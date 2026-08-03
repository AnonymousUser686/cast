// cast_sim — a behavioral simulator for cast programs.
//
// This is NOT the real compiler. It reuses the real cast parser
// (src/Frontend/CastParser.cpp) so parsing is identical to castc, then walks
// the AST applying the SAME semantics the MLIR lowering in castLower.cpp
// implements, and prints what a `print(...)` would emit. It exists so cast
// behaviour can be checked on a machine without the LLVM + CIRCT + iverilog
// toolchain.
//
// Modelled faithfully:
//   * multiple machine instances from the instantiate block, all on one clock
//   * machine-to-machine channel wiring (`b.in <- a.out`): point-to-point,
//     decoupled by a depth-5 FIFO; a value sent on cycle T is receivable at
//     cycle T+1; if the FIFO is full the send is dropped (the generated
//     hardware ignores the ready signal on machine outputs)
//   * constant feeds (`m.port <- 42`): an always-valid stream of that value,
//     matching the patched always-valid feed registers in the lowering
//   * header receives (`s: ch -> x` / `s: x <- ch`) gate the state body: in
//     cycles with no valid data the body does not run and the state re-tries
//   * output sends (`out <- expr`) from state bodies
//   * shared scalars and arrays, constant AND dynamic indices, unrolled for
//     loops (with same-cycle read-after-write inside loop bodies), if/else,
//     goto; registers commit together at the cycle boundary
//
// NOT modelled: enums, switch, exceptions, machine compile-time parameters.
// Within one cycle, prints from different instances appear in instance
// declaration order (real hardware leaves same-edge ordering unspecified).
//
// Usage: cast_sim [--max-cycles=N] [--trace] <file.cast>

#include "CastParser.hpp"
#include "CastAST.hpp"
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ast;

namespace {

struct SimError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

uint64_t maskWidth(uint64_t v, int w) {
    if (w >= 64) return v;
    return v & ((1ULL << w) - 1ULL);
}

int typeWidth(const Type &t) {
    if (t.name == "bool") return 1;
    if (t.name == "byte" || t.name == "uint8" || t.name == "int8") return 8;
    if (t.name == "uint16" || t.name == "int16") return 16;
    if (t.name == "uint32" || t.name == "int32" || t.name == "int") return 32;
    if (t.name == "uint64" || t.name == "int64") return 64;
    return 32;
}

// A channel between a producer output and a consumer input. Mirrors the
// esi.fifo the lowering inserts: capacity 5, one-cycle latency (staged pushes
// commit at the cycle boundary), drop when full.
struct ChanQueue {
    static constexpr size_t CAP = 5;
    bool isConst = false;      // constant feed: always-valid infinite stream
    uint64_t constVal = 0;
    std::deque<uint64_t> q;
    std::vector<uint64_t> staged;
    bool driven = false;       // has a wire or constant feed attached

    bool hasData() const { return isConst || !q.empty(); }
    uint64_t pop() {
        if (isConst) return constVal;
        uint64_t v = q.front();
        q.pop_front();
        return v;
    }
    void commit() {
        for (uint64_t v : staged)
            if (q.size() < CAP) q.push_back(v);   // overflow -> dropped
        staged.clear();
    }
};

struct MachineInfo {
    MachineDecl *decl = nullptr;
    std::set<std::string> inputs, outputs;
    std::map<std::string, int> portWidth;   // channel data width, by port name
};

// One machine instance: its registers, state, and executor.
class Inst {
public:
    std::string name;
    MachineInfo *mi = nullptr;

    std::map<std::string, uint64_t> committed, next;
    std::map<std::string, int> width;
    std::map<std::string, std::vector<int64_t>> arrayDims;
    std::map<std::string, int> arrayElemWidth;
    std::set<std::string> arrays;

    std::map<std::string, int64_t> loopConsts;
    std::map<std::string, uint64_t> localBindings;   // header receives
    int forDepth = 0;

    std::vector<std::string> stateOrder;
    std::map<std::string, StateDecl *> stateByName;
    std::string state, nextState;

    std::map<std::string, ChanQueue *> inQ;    // input port  -> queue
    std::map<std::string, ChanQueue *> outW;   // output port -> dest queue

    std::vector<std::string> *out = nullptr;   // shared print sink

    void setup() {
        MachineDecl *m = mi->decl;
        if (m->sharedBlock)
            for (auto &v : m->sharedBlock->varDecls) declareShared(*v);
        if (m->statesBlock)
            for (auto &sd : m->statesBlock->stateDecls) {
                stateOrder.push_back(sd->name);
                stateByName[sd->name] = sd.get();
            }
        if (stateOrder.empty())
            throw SimError("machine '" + mi->decl->name + "' has no states");
        state = stateOrder.front();
    }

    // Runs one cycle. Returns true if anything observable happened (body ran
    // or state changed) — used for idle detection.
    bool cycle() {
        next = committed;
        nextState = state;
        localBindings.clear();
        forDepth = 0;

        StateDecl *sd = stateByName.at(state);

        // Header receives gate the body: all channels must have data.
        std::vector<std::pair<std::string, ChanQueue *>> recvs;
        for (auto &h : sd->headerReceives) {
            std::string var, chan;
            if (!headerBinding(h.get(), var, chan))
                throw SimError("unsupported state header in machine '" +
                               mi->decl->name + "'");
            auto it = inQ.find(chan);
            if (it == inQ.end())
                throw SimError("state header receives from '" + chan +
                               "' which is not an input of machine '" +
                               mi->decl->name + "'");
            if (!it->second->hasData())
                return false;   // stall: body does not run this cycle
            recvs.push_back({var, it->second});
        }
        for (auto &[var, qq] : recvs)
            localBindings[var] = qq->pop();

        if (sd->body)
            for (auto &s : sd->body->statements) exec(s.get());
        return true;
    }

    void commit() {
        committed = next;
        state = nextState;
    }

private:
    bool headerBinding(BinaryStmt *h, std::string &var, std::string &chan) {
        auto identOf = [](Expr *e, std::string &out_) {
            if (e && e->getKind() == ASTNodeKind::IdentExpr) {
                out_ = static_cast<IdentExpr *>(e)->name;
                return true;
            }
            return false;
        };
        if (h->op == "<-")   // x <- ch
            return identOf(h->lhs.get(), var) && identOf(h->rhs.get(), chan);
        if (h->op == "->")   // ch -> x
            return identOf(h->lhs.get(), chan) && identOf(h->rhs.get(), var);
        return false;
    }

    void declareShared(VarDecl &v) {
        if (v.typedIdent.type.kind == TypeKind::ARRAY) {
            std::vector<int64_t> dims;
            const Type *t = &v.typedIdent.type;
            while (t->kind == TypeKind::ARRAY && t->elementType) {
                dims.push_back(t->size);
                t = t->elementType.get();
            }
            int ew = typeWidth(*t);
            for (auto &id : v.typedIdent.idents) {
                arrays.insert(id);
                arrayDims[id] = dims;
                arrayElemWidth[id] = ew;
                int64_t total = 1;
                for (int64_t d : dims) total *= d;
                std::vector<int64_t> idx(dims.size(), 0);
                for (int64_t f = 0; f < total; ++f) {
                    std::string key = elemKey(id, idx);
                    committed[key] = 0;
                    width[key] = ew;
                    for (int d = (int)dims.size() - 1; d >= 0; --d) {
                        if (++idx[d] < dims[d]) break;
                        idx[d] = 0;
                    }
                }
            }
            return;
        }
        int w = typeWidth(v.typedIdent.type);
        int64_t init = 0;
        if (v.initExpr) init = evalConst(v.initExpr.get());
        for (auto &id : v.typedIdent.idents) {
            committed[id] = maskWidth((uint64_t)init, w);
            width[id] = w;
        }
    }

    static std::string elemKey(const std::string &name,
                               const std::vector<int64_t> &idx) {
        std::string k = name;
        for (int64_t i : idx) k += "[" + std::to_string(i) + "]";
        return k;
    }

    // Mirrors readVar(): inside a for-loop, reads see this cycle's pending
    // writes; otherwise the start-of-cycle value.
    uint64_t readKey(const std::string &key) {
        if (forDepth > 0) {
            auto it = next.find(key);
            if (it != next.end()) return it->second;
        }
        auto it = committed.find(key);
        return it == committed.end() ? 0 : it->second;
    }

    bool indexChain(Expr *e, std::string &name, std::vector<int64_t> &idx) {
        std::vector<Expr *> ixs;
        while (e && e->getKind() == ASTNodeKind::IndexExpr) {
            auto *ix = static_cast<IndexExpr *>(e);
            ixs.insert(ixs.begin(), ix->index.get());
            e = ix->base.get();
        }
        if (!e || e->getKind() != ASTNodeKind::IdentExpr) return false;
        name = static_cast<IdentExpr *>(e)->name;
        for (auto *ie : ixs) idx.push_back((int64_t)eval(ie));
        return true;
    }

    bool inBounds(const std::string &name, const std::vector<int64_t> &idx) {
        auto &dims = arrayDims[name];
        if (idx.size() != dims.size()) return false;
        for (size_t d = 0; d < dims.size(); ++d)
            if (idx[d] < 0 || idx[d] >= dims[d]) return false;
        return true;
    }

    // Width an expression's value carries, mirroring the lowering: literals
    // default to 32 bits, names use their declared width, binary ops widen to
    // the larger operand, comparisons yield 1 bit. Needed so that negative
    // intermediates truncate the same way the hardware does -- otherwise
    // `x == 0 - 15` compares a 64-bit -15 against a 32-bit one.
    int exprWidth(Expr *e) {
        switch (e->getKind()) {
        case ASTNodeKind::NumberLiteral:
            return 32;
        case ASTNodeKind::IdentExpr: {
            std::string n = static_cast<IdentExpr *>(e)->name;
            if (loopConsts.count(n)) return 32;
            if (localBindings.count(n)) {
                auto p = mi->portWidth.find(n);
                return p == mi->portWidth.end() ? 32 : p->second;
            }
            auto it = width.find(n);
            return it == width.end() ? 32 : it->second;
        }
        case ASTNodeKind::IndexExpr: {
            Expr *b = e;
            while (b->getKind() == ASTNodeKind::IndexExpr)
                b = static_cast<IndexExpr *>(b)->base.get();
            if (b->getKind() == ASTNodeKind::IdentExpr) {
                auto it =
                    arrayElemWidth.find(static_cast<IdentExpr *>(b)->name);
                if (it != arrayElemWidth.end()) return it->second;
            }
            return 32;
        }
        case ASTNodeKind::UnaryExpr:
            return exprWidth(static_cast<UnaryExpr *>(e)->expr.get());
        case ASTNodeKind::UpdateExpr:
            return exprWidth(static_cast<UpdateExpr *>(e)->expr.get());
        case ASTNodeKind::BinaryExpr: {
            auto *b = static_cast<BinaryExpr *>(e);
            const std::string &o = b->op;
            if (o == "==" || o == "!=" || o == "<" || o == "<=" || o == ">" ||
                o == ">=" || o == "&&" || o == "||")
                return 1;
            int lw = exprWidth(b->lhs.get()), rw = exprWidth(b->rhs.get());
            return lw > rw ? lw : rw;
        }
        default:
            return 32;
        }
    }

    uint64_t eval(Expr *e) {
        switch (e->getKind()) {
        case ASTNodeKind::NumberLiteral:
            return (uint64_t)static_cast<NumberLiteral *>(e)->value;
        case ASTNodeKind::IdentExpr: {
            std::string n = static_cast<IdentExpr *>(e)->name;
            if (loopConsts.count(n)) return (uint64_t)loopConsts[n];
            if (localBindings.count(n)) return localBindings[n];
            return readKey(n);
        }
        case ASTNodeKind::IndexExpr: {
            std::string name;
            std::vector<int64_t> idx;
            if (!indexChain(e, name, idx) || !arrays.count(name))
                throw SimError("bad array access");
            if (!inBounds(name, idx)) return 0;
            return readKey(elemKey(name, idx));
        }
        case ASTNodeKind::UnaryExpr: {
            auto *u = static_cast<UnaryExpr *>(e);
            uint64_t v = eval(u->expr.get());
            int w = exprWidth(u->expr.get());
            if (u->op == "-") return maskWidth((uint64_t)(-(int64_t)v), w);
            if (u->op == "!") return v == 0 ? 1 : 0;
            throw SimError("unary '" + u->op + "' not modelled");
        }
        case ASTNodeKind::BinaryExpr: {
            auto *b = static_cast<BinaryExpr *>(e);
            const std::string &o = b->op;
            // Both operands are widened to the same width and every result is
            // truncated to it, matching comb.* op semantics in the lowering.
            int lw = exprWidth(b->lhs.get()), rw = exprWidth(b->rhs.get());
            int w = lw > rw ? lw : rw;
            uint64_t l = maskWidth(eval(b->lhs.get()), w);
            uint64_t r = maskWidth(eval(b->rhs.get()), w);
            if (o == "+") return maskWidth(l + r, w);
            if (o == "-") return maskWidth(l - r, w);
            if (o == "*") return maskWidth(l * r, w);
            if (o == "/") return r ? maskWidth(l / r, w) : 0;
            if (o == "%") return r ? maskWidth(l % r, w) : 0;
            if (o == "&") return maskWidth(l & r, w);
            if (o == "|") return maskWidth(l | r, w);
            if (o == "^") return maskWidth(l ^ r, w);
            if (o == "<<") return maskWidth(r >= 64 ? 0 : l << r, w);
            if (o == ">>") return r >= 64 ? 0 : l >> r;   // logical, as ShrU
            if (o == "==") return l == r;
            if (o == "!=") return l != r;
            if (o == "<") return l < r;                   // unsigned, as ICmp ult
            if (o == "<=") return l <= r;
            if (o == ">") return l > r;
            if (o == ">=") return l >= r;
            if (o == "&&") return (l && r) ? 1 : 0;
            if (o == "||") return (l || r) ? 1 : 0;
            throw SimError("binary '" + o + "' not modelled");
        }
        case ASTNodeKind::UpdateExpr:
            return eval(static_cast<UpdateExpr *>(e)->expr.get());
        default:
            throw SimError("expression kind not modelled");
        }
    }

    int64_t evalConst(Expr *e) {
        switch (e->getKind()) {
        case ASTNodeKind::NumberLiteral:
            return static_cast<NumberLiteral *>(e)->value;
        case ASTNodeKind::IdentExpr: {
            std::string n = static_cast<IdentExpr *>(e)->name;
            if (loopConsts.count(n)) return loopConsts[n];
            throw SimError("for-loop bound references runtime value '" + n +
                           "' — not a compile-time constant");
        }
        case ASTNodeKind::UnaryExpr: {
            auto *u = static_cast<UnaryExpr *>(e);
            int64_t v = evalConst(u->expr.get());
            return u->op == "-" ? -v : (u->op == "!" ? !v : v);
        }
        case ASTNodeKind::BinaryExpr: {
            auto *b = static_cast<BinaryExpr *>(e);
            int64_t l = evalConst(b->lhs.get()), r = evalConst(b->rhs.get());
            const std::string &o = b->op;
            if (o == "+") return l + r;
            if (o == "-") return l - r;
            if (o == "*") return l * r;
            if (o == "/") return r ? l / r : 0;
            if (o == "%") return r ? l % r : 0;
            if (o == "<<") return l << r;
            if (o == ">>") return l >> r;
            if (o == "&") return l & r;
            if (o == "|") return l | r;
            if (o == "^") return l ^ r;
            if (o == "<") return l < r;
            if (o == "<=") return l <= r;
            if (o == ">") return l > r;
            if (o == ">=") return l >= r;
            if (o == "==") return l == r;
            if (o == "!=") return l != r;
            throw SimError("non-constant for-loop expression");
        }
        default:
            throw SimError("for-loop control must be compile-time constant");
        }
    }

    void writeScalar(const std::string &name, const std::string &op,
                     uint64_t rhs) {
        int w = width.count(name) ? width[name] : 32;
        uint64_t cur = readKey(name);
        next[name] = maskWidth(applyOp(op, cur, rhs), w);
    }

    void writeElem(Expr *lhs, const std::string &op, uint64_t rhs) {
        std::string name;
        std::vector<int64_t> idx;
        if (!indexChain(lhs, name, idx) || !arrays.count(name))
            throw SimError("bad array write target");
        if (!inBounds(name, idx)) return;
        std::string key = elemKey(name, idx);
        int w = arrayElemWidth[name];
        uint64_t cur = readKey(key);
        next[key] = maskWidth(applyOp(op, cur, rhs), w);
    }

    static uint64_t applyOp(const std::string &op, uint64_t cur, uint64_t rhs) {
        if (op == "=" || op == ":=") return rhs;
        if (op == "+=") return cur + rhs;
        if (op == "-=") return cur - rhs;
        if (op == "*=") return cur * rhs;
        if (op == "/=") return rhs ? cur / rhs : 0;
        if (op == "%=") return rhs ? cur % rhs : 0;
        if (op == "^=") return cur ^ rhs;
        if (op == "&=") return cur & rhs;
        if (op == "|=") return cur | rhs;
        if (op == "<<=") return cur << rhs;
        if (op == ">>=") return cur >> rhs;
        return rhs;
    }

    void exec(Stmt *s) {
        switch (s->getKind()) {
        case ASTNodeKind::BlockStmt:
            for (auto &st : static_cast<BlockStmt *>(s)->statements)
                exec(st.get());
            return;
        case ASTNodeKind::VarDeclStmt: {
            auto vd = static_cast<VarDeclStmt *>(s)->varDecl;
            if (vd->typedIdent.type.kind != TypeKind::ARRAY) {
                int w = typeWidth(vd->typedIdent.type);
                for (auto &id : vd->typedIdent.idents) {
                    if (!width.count(id)) { width[id] = w; committed[id] = 0; }
                    if (vd->initExpr)
                        writeScalar(id, "=", eval(vd->initExpr.get()));
                }
            }
            return;
        }
        case ASTNodeKind::BinaryStmt: {
            auto *b = static_cast<BinaryStmt *>(s);
            if (b->op == "<-" || b->op == "->") {
                // Output send: `out <- expr` or `expr -> out`.
                bool rev = (b->op == "->");
                Expr *dstE = rev ? b->rhs.get() : b->lhs.get();
                Expr *srcE = rev ? b->lhs.get() : b->rhs.get();
                if (dstE->getKind() == ASTNodeKind::IdentExpr) {
                    std::string port =
                        static_cast<IdentExpr *>(dstE)->name;
                    if (mi->outputs.count(port)) {
                        uint64_t v = eval(srcE);
                        auto it = outW.find(port);
                        if (it != outW.end())
                            it->second->staged.push_back(v);
                        // unconnected outputs: value goes nowhere (as in HW)
                        return;
                    }
                }
                return;   // other channel forms in bodies: not modelled
            }
            uint64_t rhs = eval(b->rhs.get());
            if (b->lhs->getKind() == ASTNodeKind::IndexExpr)
                writeElem(b->lhs.get(), b->op, rhs);
            else if (b->lhs->getKind() == ASTNodeKind::IdentExpr)
                writeScalar(static_cast<IdentExpr *>(b->lhs.get())->name,
                            b->op, rhs);
            return;
        }
        case ASTNodeKind::ExprStmt: {
            auto *e = static_cast<ExprStmt *>(s)->expr.get();
            if (e->getKind() == ASTNodeKind::UpdateExpr)
                execUpdate(static_cast<UpdateExpr *>(e));
            else if (e->getKind() == ASTNodeKind::CallExpr)
                execCall(static_cast<CallExpr *>(e));
            return;
        }
        case ASTNodeKind::IfStmt: {
            auto *i = static_cast<IfStmt *>(s);
            if (eval(i->cond.get())) {
                if (i->thenBranch) exec(i->thenBranch.get());
            } else if (i->elseBranch) {
                exec(i->elseBranch.get());
            }
            return;
        }
        case ASTNodeKind::ForStmt:
            execFor(static_cast<ForStmt *>(s));
            return;
        case ASTNodeKind::GotoStmt:
            nextState = static_cast<GotoStmt *>(s)->targetState;
            return;
        default:
            return;
        }
    }

    void execUpdate(UpdateExpr *u) {
        std::string op = (u->op == "--") ? "-=" : "+=";
        if (u->expr->getKind() == ASTNodeKind::IndexExpr)
            writeElem(u->expr.get(), op, 1);
        else if (u->expr->getKind() == ASTNodeKind::IdentExpr)
            writeScalar(static_cast<IdentExpr *>(u->expr.get())->name, op, 1);
    }

    void execFor(ForStmt *s) {
        std::string var;
        int64_t cur = 0;
        if (s->init && s->init->getKind() == ASTNodeKind::VarDeclStmt) {
            auto vd = static_cast<VarDeclStmt *>(s->init.get())->varDecl;
            var = vd->typedIdent.idents.empty() ? "" : vd->typedIdent.idents[0];
            if (vd->initExpr) cur = evalConst(vd->initExpr.get());
        } else if (s->init && s->init->getKind() == ASTNodeKind::BinaryStmt) {
            auto *bs = static_cast<BinaryStmt *>(s->init.get());
            if (bs->lhs->getKind() == ASTNodeKind::IdentExpr)
                var = static_cast<IdentExpr *>(bs->lhs.get())->name;
            cur = evalConst(bs->rhs.get());
        }
        if (var.empty()) throw SimError("for-loop needs a loop variable");

        bool hadOuter = loopConsts.count(var);
        int64_t outer = hadOuter ? loopConsts[var] : 0;

        ++forDepth;
        int guard = 0;
        while (true) {
            if (++guard > 1000000)
                throw SimError("for-loop exceeded 1e6 unrolls");
            loopConsts[var] = cur;
            if (s->cond && !evalConst(s->cond.get())) break;
            if (!s->cond) throw SimError("for-loop needs a bound");
            if (s->body) exec(s->body.get());
            cur = applyForUpdate(s->update.get(), var, cur);
        }
        --forDepth;

        if (hadOuter) loopConsts[var] = outer; else loopConsts.erase(var);
        if (width.count(var))
            next[var] = maskWidth((uint64_t)cur, width[var]);
    }

    int64_t applyForUpdate(Stmt *u, const std::string &var, int64_t cur) {
        if (!u) throw SimError("for-loop needs an update");
        if (u->getKind() == ASTNodeKind::ExprStmt) {
            auto *e = static_cast<ExprStmt *>(u)->expr.get();
            if (e->getKind() == ASTNodeKind::UpdateExpr) {
                auto *ue = static_cast<UpdateExpr *>(e);
                return ue->op == "--" ? cur - 1 : cur + 1;
            }
        } else if (u->getKind() == ASTNodeKind::BinaryStmt) {
            auto *bs = static_cast<BinaryStmt *>(u);
            int64_t r = evalConst(bs->rhs.get());
            const std::string &o = bs->op;
            if (o == "=" || o == ":=") return r;
            if (o == "+=") return cur + r;
            if (o == "-=") return cur - r;
            if (o == "*=") return cur * r;
            if (o == "/=") return r ? cur / r : cur;
            if (o == "<<=") return cur << r;
            if (o == ">>=") return cur >> r;
        }
        throw SimError("unsupported for-loop update on '" + var + "'");
    }

    void execCall(CallExpr *c) {
        if (c->funcName != "print") return;
        std::string line;
        for (auto &a : c->args) {
            if (a->getKind() == ASTNodeKind::StringLiteral)
                line += static_cast<StringLiteral *>(a.get())->value;
            else
                line += std::to_string(eval(a.get()));
        }
        out->push_back(line);
    }
};

class Sim {
public:
    void build(Program *prog) {
        InstantiateDecl *instBlock = nullptr;
        for (auto &d : prog->decls) {
            if (!d) continue;
            if (d->getKind() == ASTNodeKind::MachineDecl) {
                auto *m = static_cast<MachineDecl *>(d.get());
                MachineInfo &info = machines[m->name];
                info.decl = m;
                if (m->interfaceBlock)
                    for (auto &io : m->interfaceBlock->ioDecls)
                        for (auto &id : io.idents) {
                            (io.direction == "input" ? info.inputs
                                                     : info.outputs)
                                .insert(id);
                            info.portWidth[id] = typeWidth(io.typedIdent.type);
                        }
            } else if (d->getKind() == ASTNodeKind::InstantiateDecl) {
                instBlock = static_cast<InstantiateDecl *>(d.get());
            }
        }
        if (!instBlock || !instBlock->body)
            throw SimError("no instantiate block found");

        // First pass: create instances.
        for (auto &s : instBlock->body->statements) {
            if (s->getKind() != ASTNodeKind::InstModuleStmt) continue;
            auto *im = static_cast<InstModuleStmt *>(s.get());
            auto mIt = machines.find(im->callExpr->funcName);
            if (mIt == machines.end())
                throw SimError("machine '" + im->callExpr->funcName +
                               "' not found for instantiation");
            auto inst = std::make_unique<Inst>();
            inst->name = im->varName;
            inst->mi = &mIt->second;
            inst->out = &out;
            inst->setup();
            // one queue per input port (unfed = never valid)
            for (auto &p : mIt->second.inputs) {
                queues.push_back(std::make_unique<ChanQueue>());
                inst->inQ[p] = queues.back().get();
            }
            byName[im->varName] = inst.get();
            insts.push_back(std::move(inst));
        }

        // Second pass: wires and constant feeds.
        for (auto &s : instBlock->body->statements) {
            if (s->getKind() != ASTNodeKind::BinaryStmt) continue;
            auto *b = static_cast<BinaryStmt *>(s.get());
            if (b->op != "<-" && b->op != "->") continue;
            bool rev = (b->op == "->");
            Expr *dstE = rev ? b->rhs.get() : b->lhs.get();
            Expr *srcE = rev ? b->lhs.get() : b->rhs.get();

            if (dstE->getKind() != ASTNodeKind::FieldExpr)
                throw SimError("instantiate feeds must target <inst>.<port>");
            auto *df = static_cast<FieldExpr *>(dstE);
            Inst *dst = byName.count(df->object) ? byName[df->object] : nullptr;
            if (!dst)
                throw SimError("unknown instance '" + df->object + "'");
            auto qIt = dst->inQ.find(df->field);
            if (qIt == dst->inQ.end())
                throw SimError("instance '" + df->object +
                               "' has no input channel '" + df->field + "'");
            ChanQueue *q = qIt->second;
            if (q->driven)
                throw SimError("input '" + df->object + "." + df->field +
                               "' is already driven");

            if (srcE->getKind() == ASTNodeKind::FieldExpr) {
                auto *sf = static_cast<FieldExpr *>(srcE);
                Inst *src =
                    byName.count(sf->object) ? byName[sf->object] : nullptr;
                if (!src)
                    throw SimError("unknown instance '" + sf->object + "'");
                if (!src->mi->outputs.count(sf->field))
                    throw SimError("instance '" + sf->object +
                                   "' has no output channel '" + sf->field +
                                   "'");
                if (src->outW.count(sf->field))
                    throw SimError("output '" + sf->object + "." + sf->field +
                                   "' is already connected (channels are "
                                   "point-to-point)");
                src->outW[sf->field] = q;
                q->driven = true;
            } else if (srcE->getKind() == ASTNodeKind::NumberLiteral ||
                       srcE->getKind() == ASTNodeKind::UnaryExpr) {
                // constant feed → always-valid stream
                int64_t v = constOf(srcE);
                q->isConst = true;
                q->constVal = (uint64_t)v;
                q->driven = true;
            } else {
                throw SimError(
                    "instantiate feed must be a literal or <inst>.<port> "
                    "(enums are not modelled by cast_sim)");
            }
        }
        if (insts.empty()) throw SimError("no instances created");
    }

    void run(int maxCycles, bool trace) {
        int idle = 0;
        for (int cycle = 0; cycle < maxCycles; ++cycle) {
            size_t outBefore = out.size();
            if (trace) {
                std::string line = "--- cycle " + std::to_string(cycle) + " :";
                for (auto &i : insts) line += " " + i->name + "=" + i->state;
                out.push_back(line + " ---");
            }

            bool anyActive = false;
            for (auto &i : insts)
                if (i->cycle()) anyActive = true;

            bool queuesMoving = false;
            for (auto &q : queues) {
                if (!q->staged.empty()) queuesMoving = true;
                q->commit();
            }
            bool stateChanged = false, regsChanged = false;
            for (auto &i : insts) {
                if (i->nextState != i->state) stateChanged = true;
                if (i->next != i->committed) regsChanged = true;
                i->commit();
            }
            bool printed = out.size() > outBefore + (trace ? 1 : 0);
            (void)anyActive;

            if (!printed && !stateChanged && !regsChanged && !queuesMoving) {
                if (++idle >= 2) break;   // whole system is quiescent
            } else {
                idle = 0;
            }
        }
        for (auto &line : out) std::cout << line << "\n";
    }

private:
    static int64_t constOf(Expr *e) {
        if (e->getKind() == ASTNodeKind::NumberLiteral)
            return static_cast<NumberLiteral *>(e)->value;
        if (e->getKind() == ASTNodeKind::UnaryExpr) {
            auto *u = static_cast<UnaryExpr *>(e);
            if (u->op == "-") return -constOf(u->expr.get());
        }
        throw SimError("unsupported constant feed expression");
    }

    std::map<std::string, MachineInfo> machines;
    std::vector<std::unique_ptr<Inst>> insts;
    std::map<std::string, Inst *> byName;
    std::vector<std::unique_ptr<ChanQueue>> queues;
    std::vector<std::string> out;
};

} // namespace

int main(int argc, char **argv) {
    std::string file;
    int maxCycles = 300;
    bool trace = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--max-cycles=", 0) == 0) maxCycles = std::stoi(a.substr(13));
        else if (a == "--trace") trace = true;
        else if (a.rfind("--", 0) == 0) { std::cerr << "unknown flag " << a << "\n"; return 2; }
        else file = a;
    }
    if (file.empty()) {
        std::cerr << "usage: cast_sim [--max-cycles=N] [--trace] <file.cast>\n";
        return 2;
    }
    std::ifstream in(file);
    if (!in) { std::cerr << "cannot open " << file << "\n"; return 2; }
    std::stringstream ss; ss << in.rdbuf();

    try {
        CastParser parser(ss.str());
        auto prog = parser.parseProgram();
        Sim sim;
        sim.build(prog.get());
        sim.run(maxCycles, trace);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "sim error: " << e.what() << "\n";
        return 1;
    }
}
