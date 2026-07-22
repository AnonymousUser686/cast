// cast_sim — a behavioral simulator for cast programs.
//
// This is NOT the real compiler. It reuses the real cast parser
// (src/Frontend/CastParser.cpp) so parsing is identical to castc, then walks
// the AST applying the SAME semantics the MLIR lowering in castLower.cpp
// implements, and prints what a `print(...)` would emit. It exists so the
// array / for-loop behaviour can be checked on a machine that does not have
// the LLVM + CIRCT + iverilog toolchain installed.
//
// Modelled faithfully:
//   * shared scalars and arrays (one register per element, "a[i][j]" keys)
//   * constant AND dynamic array indices (out-of-range dyn read = 0, write = nop)
//   * compile-time-unrolled for loops (bounds must be constants)
//   * read-after-write: inside a for-loop body reads see writes made earlier
//     in the same cycle; outside loops reads return the start-of-cycle value
//     (registers commit together at the cycle end) — matches readVar()
//   * if / else-if / else, taken-path execution
//   * one active state per cycle, goto selects the next state
//
// NOT modelled (not needed for the array/loop examples): interface channels,
// instantiate wiring, enums, exceptions, switch. Programs using those should
// be checked with the real toolchain.
//
// Usage: cast_sim [--max-cycles=N] [--trace] <file.cast>

#include "CastParser.hpp"
#include "CastAST.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
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

class Sim {
public:
    explicit Sim(MachineDecl *m) : machine(m) {}

    void run(int maxCycles, bool trace) {
        setup();
        if (stateOrder.empty()) throw SimError("machine has no states");

        std::string cur = stateOrder.front();
        int idle = 0;
        for (int cycle = 0; cycle < maxCycles; ++cycle) {
            next = committed;                 // registers hold, then update
            nextState = cur;
            size_t outBefore = out.size();

            if (trace) out.push_back("--- cycle " + std::to_string(cycle) +
                                     " : state " + cur + " ---");

            forDepth = 0;
            StateDecl *sd = stateByName.at(cur);
            if (sd->body)
                for (auto &s : sd->body->statements) exec(s.get());

            committed = next;
            bool producedOutput = out.size() > outBefore + (trace ? 1 : 0);
            bool stayed = (nextState == cur);
            cur = nextState;

            // Stop once we settle into a self-looping state that prints nothing
            // (the conventional `halt: { goto halt; }`).
            if (stayed && !producedOutput) {
                if (++idle >= 1) break;
            } else {
                idle = 0;
            }
        }

        for (auto &line : out) std::cout << line << "\n";
    }

private:
    MachineDecl *machine;

    std::map<std::string, uint64_t> committed, next;
    std::map<std::string, int> width;              // per scalar / element key
    std::map<std::string, std::vector<int64_t>> arrayDims;
    std::map<std::string, int> arrayElemWidth;
    std::set<std::string> arrays;

    std::map<std::string, int64_t> loopConsts;
    int forDepth = 0;

    std::vector<std::string> stateOrder;
    std::map<std::string, StateDecl *> stateByName;
    std::string nextState;

    std::vector<std::string> out;

    // ── setup ────────────────────────────────────────────────────────────────
    void setup() {
        if (machine->sharedBlock)
            for (auto &v : machine->sharedBlock->varDecls) declareShared(*v);
        if (machine->statesBlock)
            for (auto &sd : machine->statesBlock->stateDecls) {
                stateOrder.push_back(sd->name);
                stateByName[sd->name] = sd.get();
            }
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

    // ── reads ────────────────────────────────────────────────────────────────
    // Mirrors readVar(): inside a for-loop return the pending (this-cycle) value
    // so unrolled accumulations work; otherwise return the committed value.
    uint64_t readKey(const std::string &key) {
        if (forDepth > 0) {
            auto it = next.find(key);
            if (it != next.end()) return it->second;
        }
        auto it = committed.find(key);
        return it == committed.end() ? 0 : it->second;
    }

    // Resolve base name + concrete index values of an a[i][j] chain.
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

    uint64_t eval(Expr *e) {
        switch (e->getKind()) {
        case ASTNodeKind::NumberLiteral:
            return (uint64_t)static_cast<NumberLiteral *>(e)->value;
        case ASTNodeKind::IdentExpr: {
            std::string n = static_cast<IdentExpr *>(e)->name;
            if (loopConsts.count(n)) return (uint64_t)loopConsts[n];
            return readKey(n);
        }
        case ASTNodeKind::IndexExpr: {
            std::string name;
            std::vector<int64_t> idx;
            if (!indexChain(e, name, idx) || !arrays.count(name))
                throw SimError("bad array access");
            if (!inBounds(name, idx)) return 0;   // dynamic OOB read = 0
            return readKey(elemKey(name, idx));
        }
        case ASTNodeKind::UnaryExpr: {
            auto *u = static_cast<UnaryExpr *>(e);
            uint64_t v = eval(u->expr.get());
            if (u->op == "-") return (uint64_t)(-(int64_t)v);
            if (u->op == "!") return v == 0 ? 1 : 0;
            throw SimError("unary '" + u->op + "' not modelled");
        }
        case ASTNodeKind::BinaryExpr: {
            auto *b = static_cast<BinaryExpr *>(e);
            uint64_t l = eval(b->lhs.get()), r = eval(b->rhs.get());
            const std::string &o = b->op;
            if (o == "+") return l + r;
            if (o == "-") return l - r;
            if (o == "*") return l * r;
            if (o == "/") return r ? l / r : 0;
            if (o == "%") return r ? l % r : 0;
            if (o == "&") return l & r;
            if (o == "|") return l | r;
            if (o == "^") return l ^ r;
            if (o == "<<") return l << r;
            if (o == ">>") return l >> r;
            if (o == "==") return l == r;
            if (o == "!=") return l != r;
            if (o == "<") return l < r;
            if (o == "<=") return l <= r;
            if (o == ">") return l > r;
            if (o == ">=") return l >= r;
            if (o == "&&") return (l && r) ? 1 : 0;
            if (o == "||") return (l || r) ? 1 : 0;
            throw SimError("binary '" + o + "' not modelled");
        }
        case ASTNodeKind::UpdateExpr: {
            // value of x++ is x; the write is applied by execUpdate
            auto *u = static_cast<UpdateExpr *>(e);
            return eval(u->expr.get());
        }
        default:
            throw SimError("expression kind not modelled");
        }
    }

    // Compile-time evaluation for loop bounds/updates. Registers are rejected
    // exactly like the real compiler rejects runtime for-bounds.
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

    // ── writes ─────────────────────────────────────────────────────────────
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
        if (!inBounds(name, idx)) return;         // dynamic OOB write = nop
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

    // ── statements ───────────────────────────────────────────────────────────
    void exec(Stmt *s) {
        switch (s->getKind()) {
        case ASTNodeKind::BlockStmt:
            for (auto &st : static_cast<BlockStmt *>(s)->statements) exec(st.get());
            return;
        case ASTNodeKind::VarDeclStmt: {
            // A local var becomes a register too; only meaningful if later read.
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
            if (b->op == "<-" || b->op == "->") return;   // channels: not modelled
            uint64_t rhs = eval(b->rhs.get());
            if (b->lhs->getKind() == ASTNodeKind::IndexExpr)
                writeElem(b->lhs.get(), b->op, rhs);
            else if (b->lhs->getKind() == ASTNodeKind::IdentExpr)
                writeScalar(static_cast<IdentExpr *>(b->lhs.get())->name, b->op, rhs);
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
        case ASTNodeKind::IfStmt:
            execIf(static_cast<IfStmt *>(s));
            return;
        case ASTNodeKind::ForStmt:
            execFor(static_cast<ForStmt *>(s));
            return;
        case ASTNodeKind::GotoStmt:
            nextState = static_cast<GotoStmt *>(s)->targetState;
            return;
        default:
            return;   // switch/return etc. — not used by the array/loop examples
        }
    }

    void execUpdate(UpdateExpr *u) {
        std::string op = (u->op == "--") ? "-=" : "+=";
        if (u->expr->getKind() == ASTNodeKind::IndexExpr)
            writeElem(u->expr.get(), op, 1);
        else if (u->expr->getKind() == ASTNodeKind::IdentExpr)
            writeScalar(static_cast<IdentExpr *>(u->expr.get())->name, op, 1);
    }

    void execIf(IfStmt *s) {
        if (eval(s->cond.get())) {
            if (s->thenBranch) exec(s->thenBranch.get());
        } else if (s->elseBranch) {
            exec(s->elseBranch.get());
        }
    }

    void execFor(ForStmt *s) {
        std::string var;
        int64_t cur = 0;
        // init
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
            if (++guard > 1000000) throw SimError("for-loop exceeded 1e6 unrolls");
            loopConsts[var] = cur;
            if (s->cond && !evalConst(s->cond.get())) break;
            if (!s->cond) throw SimError("for-loop needs a bound");
            if (s->body) exec(s->body.get());
            cur = applyForUpdate(s->update.get(), var, cur);
        }
        --forDepth;

        if (hadOuter) loopConsts[var] = outer; else loopConsts.erase(var);
        // If the loop variable is a shared register, commit its final value.
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
        out.push_back(line);
    }
};

} // namespace

int main(int argc, char **argv) {
    std::string file;
    int maxCycles = 200;
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
        MachineDecl *machine = nullptr;
        for (auto &d : prog->decls)
            if (d && d->getKind() == ASTNodeKind::MachineDecl) {
                machine = static_cast<MachineDecl *>(d.get());
                break;
            }
        if (!machine) { std::cerr << "no machine found\n"; return 1; }
        Sim sim(machine);
        sim.run(maxCycles, trace);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "sim error: " << e.what() << "\n";
        return 1;
    }
}
