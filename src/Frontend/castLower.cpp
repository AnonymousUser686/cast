#include "castLower.hpp"
#include <iostream>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include "mlir/IR/MLIRContext.h"
#include <circt/Dialect/HW/HWTypes.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/ESI/ESIOps.h>
#include <circt/Dialect/ESI/ESITypes.h>
#include <circt/Dialect/FSM/FSMOps.h>
#include <circt/Dialect/SV/SVOps.h>
#include <circt/Dialect/SV/SVTypes.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <circt/Dialect/SV/SVAttributes.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/Seq/SeqDialect.h>
#include <circt/Dialect/Seq/SeqOps.h>
#include "llvm/Support/raw_ostream.h"

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// Error helpers
// ─────────────────────────────────────────────────────────────────────────────

[[noreturn]] static void notImplemented(const std::string &construct)
{
    std::cerr << "Error: '" << construct << "' is not yet supported by the cast compiler.\n";
    exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// AST Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visit(ast::ASTNode *node)
{
    if (!node) return std::any();
    switch (node->getKind())
    {
    case ast::ASTNodeKind::Program:
        for (auto &decl : static_cast<ast::Program*>(node)->decls)
        {
            visit(decl.get());
        }
        return std::any();
    case ast::ASTNodeKind::EnumDecl:
        return visitDecl_enum(*static_cast<ast::EnumDecl*>(node));
    case ast::ASTNodeKind::MachineDecl:
        return visitDecl_machine(*static_cast<ast::MachineDecl*>(node));
    case ast::ASTNodeKind::InterfaceBlock:
        return visitDecl_interface(*static_cast<ast::InterfaceBlock*>(node));
    case ast::ASTNodeKind::SharedBlock:
        return visitDecl_shared(*static_cast<ast::SharedBlock*>(node));
    case ast::ASTNodeKind::StatesBlock:
        return visitDecl_states(*static_cast<ast::StatesBlock*>(node));
    case ast::ASTNodeKind::StateDecl:
        return visitDecl_state(*static_cast<ast::StateDecl*>(node));
    case ast::ASTNodeKind::InstantiateDecl:
        return visitDecl_instantiate(*static_cast<ast::InstantiateDecl*>(node));
    case ast::ASTNodeKind::VarDecl:
        return visitDecl_var(*static_cast<ast::VarDecl*>(node));
    
    // Statements
    case ast::ASTNodeKind::BlockStmt:
        return visitBlockStmt(*static_cast<ast::BlockStmt*>(node));
    case ast::ASTNodeKind::VarDeclStmt:
        return visitDecl_var(*static_cast<ast::VarDeclStmt*>(node)->varDecl);
    case ast::ASTNodeKind::InstModuleStmt:
        return visitInst_module(*static_cast<ast::InstModuleStmt*>(node));
    case ast::ASTNodeKind::IfStmt:
        return visitStmt_if(*static_cast<ast::IfStmt*>(node));
    case ast::ASTNodeKind::BinaryStmt:
        return visitStmt_binary(*static_cast<ast::BinaryStmt*>(node));
    case ast::ASTNodeKind::GotoStmt:
        return visitStmt_nextstate(*static_cast<ast::GotoStmt*>(node));
    case ast::ASTNodeKind::ExprStmt:
        return visit(static_cast<ast::ExprStmt*>(node)->expr.get());
    case ast::ASTNodeKind::ForStmt:
        return visitForStmt(*static_cast<ast::ForStmt*>(node));
    case ast::ASTNodeKind::SwitchStmt:
        return visitSwitchStmt(*static_cast<ast::SwitchStmt*>(node));
    case ast::ASTNodeKind::ReturnStmt:
        return visitReturnStmt(*static_cast<ast::ReturnStmt*>(node));

    // Expressions
    case ast::ASTNodeKind::IdentExpr:
        return visitIdent(*static_cast<ast::IdentExpr*>(node));
    case ast::ASTNodeKind::FieldExpr:
        return visitIdent_field(*static_cast<ast::FieldExpr*>(node));
    case ast::ASTNodeKind::NumberLiteral:
        return visitNumber_literal(*static_cast<ast::NumberLiteral*>(node));
    case ast::ASTNodeKind::StringLiteral:
        notImplemented("string literals in expressions");
    case ast::ASTNodeKind::NilLiteral:
        notImplemented("nil literals");
    case ast::ASTNodeKind::UnaryExpr:
    case ast::ASTNodeKind::BinaryExpr:
    case ast::ASTNodeKind::UpdateExpr:
        return visitExpr(*static_cast<ast::Expr*>(node));
    case ast::ASTNodeKind::CallExpr:
        return visitExpr_func_call(*static_cast<ast::CallExpr*>(node));
    
    default:
        return std::any();
    }
}

std::any LowerVisitor::visit(const std::shared_ptr<ast::ASTNode> &node)
{
    return visit(node.get());
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

mlir::Value LowerVisitor::coerce(mlir::Value v, mlir::Type t, mlir::Location loc)
{
    if (!v || !t) return v;
    auto srcInt = mlir::dyn_cast<mlir::IntegerType>(v.getType());
    auto dstInt = mlir::dyn_cast<mlir::IntegerType>(t);
    if (!srcInt || !dstInt) return v;

    if (!srcInt.isSignless())
    {
        mlir::Type signless = mlir::IntegerType::get(&this->ctx, srcInt.getWidth());
        v = mlir::UnrealizedConversionCastOp::create(builder, loc, signless, mlir::ValueRange{v}).getResult(0);
        srcInt = mlir::cast<mlir::IntegerType>(v.getType());
    }
    mlir::Type targetSignless = dstInt.isSignless()
        ? t
        : mlir::IntegerType::get(&this->ctx, dstInt.getWidth());

    if (v.getType() == targetSignless) return v;
    if (srcInt.getWidth() == dstInt.getWidth()) return v;
    if (srcInt.getWidth() < dstInt.getWidth())
    {
        unsigned padW = dstInt.getWidth() - srcInt.getWidth();
        mlir::Value zero = circt::hw::ConstantOp::create(
            builder, loc, builder.getIntegerType(padW), 0);
        return circt::comb::ConcatOp::create(builder, loc, mlir::ValueRange{zero, v});
    }
    return circt::comb::ExtractOp::create(builder, loc, targetSignless, v, 0);
}

mlir::Value LowerVisitor::readVar(const std::string &name)
{
    auto &mvars = this->varRegs[this->currentModuleName];
    auto it = mvars.find(name);
    if (it == mvars.end()) return nullptr;
    return it->second.getResult();
}

void LowerVisitor::writeVar(const std::string &name, mlir::Value newVal, mlir::Value cond)
{
    auto &mvars = this->varRegs[this->currentModuleName];
    auto it = mvars.find(name);
    if (it == mvars.end()) return;
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value prev = this->varNext[this->currentModuleName][name];
    mlir::Value coerced = coerce(newVal, prev.getType(), loc);
    mlir::Value next = circt::comb::MuxOp::create(builder, loc, cond, coerced, prev);
    this->varNext[this->currentModuleName][name] = next;
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level declarations
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitDecl_enum(ast::EnumDecl &decl)
{
    unsigned width = 8;
    for (size_t i = 0; i < decl.members.size(); ++i)
    {
        std::string name = decl.members[i];
        this->enumValues[name] = {static_cast<uint64_t>(i), width};
    }
    return std::any();
}

std::any LowerVisitor::visitDecl_machine(ast::MachineDecl &decl)
{
    mlir::OpBuilder &b = this->builder;
    mlir::Location loc = b.getUnknownLoc();
    b.setInsertionPointToStart(this->topModule.getBody());

    this->currentPorts.clear();
    std::string moduleName = decl.name;
    this->currentModuleName = moduleName;
    this->variables[moduleName];
    this->portValues[moduleName];
    this->portDirs[moduleName];
    this->varRegs[moduleName];
    this->varNext[moduleName];
    this->stateIds[moduleName];
    this->stateActive[moduleName];
    this->outputDataNext[moduleName];
    this->outputValidNext[moduleName];

    if (decl.interfaceBlock)
        visit(decl.interfaceBlock);

    circt::hw::PortInfo rstPort;
    rstPort.name = b.getStringAttr("rst");
    rstPort.type = b.getI1Type();
    rstPort.dir = circt::hw::ModulePort::Direction::Input;
    this->currentPorts.insert(this->currentPorts.begin(), rstPort);

    circt::hw::PortInfo clkPort;
    clkPort.name = b.getStringAttr("clk");
    clkPort.type = circt::seq::ClockType::get(&this->ctx);
    clkPort.dir = circt::hw::ModulePort::Direction::Input;
    this->currentPorts.insert(this->currentPorts.begin(), clkPort);

    circt::hw::ModulePortInfo portInfo(this->currentPorts);
    mlir::StringAttr modNameAttr = b.getStringAttr(moduleName);
    circt::hw::HWModuleOp hwModule = circt::hw::HWModuleOp::create(b, loc, modNameAttr, portInfo);
    this->modules[moduleName] = hwModule;

    mlir::Block *body = &hwModule.getBody().front();
    for (auto &op : llvm::make_early_inc_range(body->getOperations()))
        if (mlir::isa<circt::hw::OutputOp>(op)) op.erase();

    this->currentClock = body->getArgument(0);
    unsigned argIdx = 0;
    for (auto &p : this->currentPorts)
    {
        std::string pname = p.name.str();
        this->portDirs[moduleName][pname] = p.dir;
        if (p.dir == circt::hw::ModulePort::Direction::Input)
        {
            this->portValues[moduleName][pname] = body->getArgument(argIdx++);
        }
    }
    b.setInsertionPointToStart(body);
    this->currentReset = body->getArgument(1);

    if (decl.sharedBlock) visit(decl.sharedBlock);
    if (decl.statesBlock) visit(decl.statesBlock);

    for (auto &kv : this->varRegs[moduleName])
    {
        const std::string &name = kv.first;
        circt::seq::CompRegOp reg = kv.second;
        mlir::Value next = this->varNext[moduleName][name];
        if (next) reg->setOperand(0, next);
    }
    if (this->stateRegs.count(moduleName))
    {
        circt::seq::CompRegOp sreg = this->stateRegs[moduleName];
        mlir::Value next = this->stateNext[moduleName];
        if (next) sreg->setOperand(0, next);
    }
    llvm::SmallVector<mlir::Value, 4> outputOperands;
    for (auto &p : this->currentPorts)
    {
        if (p.dir != circt::hw::ModulePort::Direction::Output) continue;
        std::string pname = p.name.str();
        mlir::Value data = this->outputDataNext[moduleName][pname];
        mlir::Value valid = this->outputValidNext[moduleName][pname];
        auto chTy = mlir::dyn_cast<circt::esi::ChannelType>(p.type);
        if (!data && chTy)
            data = circt::hw::ConstantOp::create(b, loc, chTy.getInner(), 0);
        if (!valid)
            valid = circt::hw::ConstantOp::create(b, loc, b.getI1Type(), 0);
        if (chTy)
        {
            circt::esi::WrapValidReadyOp wrap = circt::esi::WrapValidReadyOp::create(
                b, loc, p.type, b.getI1Type(), data, valid);
            outputOperands.push_back(wrap.getResult(0));
        }
    }
    circt::hw::OutputOp::create(b, loc, outputOperands);

    b.setInsertionPointToEnd(this->topModule.getBody());

    return std::any();
}

std::any LowerVisitor::visitDecl_interface(ast::InterfaceBlock &block)
{
    for (auto &decl : block.ioDecls)
    {
        for (auto &id : decl.idents)
        {
            circt::hw::PortInfo port;
            port.name = builder.getStringAttr(id);
            auto maybeType = getMlirType(decl.typedIdent.type);
            mlir::Type inner = maybeType.value_or(builder.getI8Type());
            port.type = circt::esi::ChannelType::get(&this->ctx, inner);
            this->channels[id] = port.type;
            if (decl.direction == "input")
            {
                port.dir = circt::hw::ModulePort::Direction::Input;
            }
            else if (decl.direction == "output")
            {
                port.dir = circt::hw::ModulePort::Direction::Output;
            }
            else
            {
                std::cerr << "Error: machine port must be 'input' or 'output'.\n";
                exit(1);
            }
            this->currentPorts.push_back(port);
        }
    }
    return std::any();
}

std::any LowerVisitor::visitDecl_shared(ast::SharedBlock &block)
{
    mlir::Location loc = builder.getUnknownLoc();
    for (auto &v : block.varDecls)
    {
        auto maybeType = getMlirType(v->typedIdent.type);
        if (!maybeType) continue;
        for (auto &id : v->typedIdent.idents)
        {
            std::string name = id;
            this->variables[this->currentModuleName][name] = maybeType.value();
            mlir::Value zero = circt::hw::ConstantOp::create(builder, loc, maybeType.value(), 0);
            circt::seq::CompRegOp reg = circt::seq::CompRegOp::create(
                builder, loc, zero, this->currentClock, this->currentReset, zero);
            this->varRegs[this->currentModuleName][name] = reg;
            this->varNext[this->currentModuleName][name] = reg.getResult();
        }
    }
    return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// State machine
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitDecl_states(ast::StatesBlock &block)
{
    mlir::Location loc = builder.getUnknownLoc();
    auto &stateDecls = block.stateDecls;
    if (stateDecls.empty()) return std::any();

    unsigned needed = 1;
    while ((1u << needed) < stateDecls.size()) ++needed;
    if (needed < 1) needed = 1;
    mlir::Type stateTy = builder.getIntegerType(needed);

    uint32_t id = 0;
    for (auto &sd : stateDecls)
    {
        std::string name = sd->name;
        this->stateIds[this->currentModuleName][name] = id;
        ++id;
    }

    mlir::Value resetVal = circt::hw::ConstantOp::create(builder, loc, stateTy, 0);
    circt::seq::CompRegOp stateReg = circt::seq::CompRegOp::create(
        builder, loc, resetVal, this->currentClock, this->currentReset, resetVal);
    this->stateRegs[this->currentModuleName] = stateReg;
    this->stateNext[this->currentModuleName] = stateReg.getResult();

    for (auto &kv : this->stateIds[this->currentModuleName])
    {
        mlir::Value sId = circt::hw::ConstantOp::create(builder, loc, stateTy, kv.second);
        mlir::Value eq = circt::comb::ICmpOp::create(
            builder, loc, circt::comb::ICmpPredicate::eq, stateReg.getResult(), sId);
        this->stateActive[this->currentModuleName][kv.first] = eq;
    }

    for (auto &sd : stateDecls)
        visit(sd);

    return std::any();
}

std::any LowerVisitor::visitDecl_state(ast::StateDecl &decl)
{
    std::string name = decl.name;
    mlir::Value active = this->stateActive[this->currentModuleName][name];
    if (!active) return std::any();

    mlir::Value prevFire = this->currentFire;
    std::string prevName = this->currentStateName;
    bool prevIn = this->inStateBody;

    this->currentFire = active;
    this->currentStateName = name;
    this->inStateBody = true;
    this->localBindings.clear();

    this->inStateHeader = true;
    for (auto &sb : decl.headerReceives)
        visit(sb);
    this->inStateHeader = false;

    if (decl.body)
        for (auto &s : decl.body->statements)
            visit(s);

    this->localBindings.clear();
    this->currentFire = prevFire;
    this->currentStateName = prevName;
    this->inStateBody = prevIn;
    return std::any();
}

std::any LowerVisitor::visitStmt_nextstate(ast::GotoStmt &stmt)
{
    std::string target = stmt.targetState;
    if (target.empty()) return std::any();

    auto &ids = this->stateIds[this->currentModuleName];
    auto it = ids.find(target);
    if (it == ids.end()) return std::any();

    mlir::Location loc = builder.getUnknownLoc();
    circt::seq::CompRegOp sreg = this->stateRegs[this->currentModuleName];
    mlir::Type stateTy = sreg.getResult().getType();
    mlir::Value targetId = circt::hw::ConstantOp::create(builder, loc, stateTy, it->second);

    mlir::Value prev = this->stateNext[this->currentModuleName];
    mlir::Value next = circt::comb::MuxOp::create(builder, loc, this->currentFire, targetId, prev);
    this->stateNext[this->currentModuleName] = next;
    return std::any();
}

std::any LowerVisitor::visitExpr_func_call(ast::CallExpr &expr)
{
    std::string funcName = expr.funcName;
    if (funcName != "print" || !this->inStateBody || !this->currentFire)
        return std::any();

    mlir::Location loc = builder.getUnknownLoc();

    std::string fmtStr;
    llvm::SmallVector<mlir::Value, 4> subs;
    for (auto &argExpr : expr.args)
    {
        if (argExpr->getKind() == ast::ASTNodeKind::StringLiteral)
        {
            std::string raw = static_cast<ast::StringLiteral*>(argExpr.get())->value;
            fmtStr += raw;
        }
        else
        {
            auto any = visit(argExpr);
            if (any.has_value() && any.type() == typeid(mlir::Value))
            {
                mlir::Value v = std::any_cast<mlir::Value>(any);
                if (v) { fmtStr += "%0d"; subs.push_back(v); }
            }
        }
    }
    fmtStr += "\n";

    mlir::Value i1Clk = circt::seq::FromClockOp::create(builder, loc, this->currentClock);
    mlir::Value fire = this->currentFire;
    mlir::StringAttr fmtAttr = builder.getStringAttr(fmtStr);
    mlir::Value fd = circt::hw::ConstantOp::create(
        builder, loc, builder.getIntegerType(32), 0x80000001LL);

    circt::sv::AlwaysFFOp::create(builder, loc,
        circt::sv::EventControl::AtPosEdge, i1Clk,
        [&]() {
            circt::sv::IfOp::create(builder, loc, fire,
                [&]() {
                    circt::sv::FWriteOp::create(builder, loc, fd, fmtAttr, subs);
                });
        });

    return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// Expression / identifier resolution
// ─────────────────────────────────────────────────────────────────────────────

std::optional<mlir::Type> LowerVisitor::getMlirType(const ast::Type &t)
{
    if (t.kind == ast::TypeKind::PRIMITIVE)
    {
        if (t.name == "int") return builder.getI32Type();
        if (t.name == "byte") return builder.getI8Type();
        if (t.name == "int32") return builder.getI32Type();
        if (t.name == "uint32") return builder.getIntegerType(32);
        if (t.name == "uint16") return builder.getIntegerType(16);
        if (t.name == "bool") return builder.getI1Type();
        if (t.name == "string") return circt::hw::StringType::get(&this->ctx);
    }
    return std::nullopt;
}

std::any LowerVisitor::visitIdent(ast::IdentExpr &expr)
{
    std::string name = expr.name;
    if (this->localBindings.count(name))
        return mlir::Value(this->localBindings[name]);
    if (this->varRegs.count(this->currentModuleName) &&
        this->varRegs[this->currentModuleName].count(name))
        return mlir::Value(this->varRegs[this->currentModuleName][name].getResult());
    if (this->portValues.count(this->currentModuleName) &&
        this->portValues[this->currentModuleName].count(name))
        return this->portValues[this->currentModuleName][name];
    if (this->enumValues.count(name))
    {
        auto [val, w] = this->enumValues[name];
        return mlir::Value(circt::hw::ConstantOp::create(
            builder, builder.getUnknownLoc(), builder.getIntegerType(w), val));
    }
    return std::any();
}

std::any LowerVisitor::visitNumber_literal(ast::NumberLiteral &expr)
{
    int64_t value = expr.value;
    mlir::Type t = this->currentExprType ? this->currentExprType : builder.getI32Type();
    if (!mlir::isa<mlir::IntegerType>(t)) t = builder.getI32Type();
    return mlir::Value(circt::hw::ConstantOp::create(
        builder, builder.getUnknownLoc(), t, value));
}

std::any LowerVisitor::visitExpr(ast::Expr &expr)
{
    mlir::Location loc = builder.getUnknownLoc();

    if (expr.getKind() == ast::ASTNodeKind::IdentExpr)
        return visitIdent(static_cast<ast::IdentExpr&>(expr));
    if (expr.getKind() == ast::ASTNodeKind::FieldExpr)
        return visitIdent_field(static_cast<ast::FieldExpr&>(expr));
    if (expr.getKind() == ast::ASTNodeKind::NumberLiteral)
        return visitNumber_literal(static_cast<ast::NumberLiteral&>(expr));
    if (expr.getKind() == ast::ASTNodeKind::StringLiteral)
        notImplemented("string literals in expressions");
    if (expr.getKind() == ast::ASTNodeKind::NilLiteral)
        notImplemented("nil literals");
    if (expr.getKind() == ast::ASTNodeKind::CallExpr)
        return visitExpr_func_call(static_cast<ast::CallExpr&>(expr));

    if (expr.getKind() == ast::ASTNodeKind::UnaryExpr)
    {
        auto &un = static_cast<ast::UnaryExpr&>(expr);
        auto inner = visit(un.expr);
        if (!inner.has_value()) return std::any();
        mlir::Value v = std::any_cast<mlir::Value>(inner);
        if (un.op == "!")
        {
            mlir::Value one = circt::hw::ConstantOp::create(builder, loc, v.getType(), 1);
            return mlir::Value(circt::comb::XorOp::create(builder, loc, v, one));
        }
        notImplemented("unary operator '" + un.op + "'");
    }

    if (expr.getKind() == ast::ASTNodeKind::BinaryExpr)
    {
        auto &bin = static_cast<ast::BinaryExpr&>(expr);
        auto la = visit(bin.lhs);
        auto ra = visit(bin.rhs);
        if (!la.has_value() || !ra.has_value()) return std::any();
        mlir::Value lhs = std::any_cast<mlir::Value>(la);
        mlir::Value rhs = std::any_cast<mlir::Value>(ra);
        if (!lhs || !rhs) return std::any();

        auto li = mlir::dyn_cast<mlir::IntegerType>(lhs.getType());
        auto ri = mlir::dyn_cast<mlir::IntegerType>(rhs.getType());
        if (li && ri && li.getWidth() != ri.getWidth())
        {
            unsigned w = std::max(li.getWidth(), ri.getWidth());
            mlir::Type wide = builder.getIntegerType(w);
            lhs = coerce(lhs, wide, loc);
            rhs = coerce(rhs, wide, loc);
        }

        std::string op = bin.op;
        if (op == "+")  return mlir::Value(circt::comb::AddOp::create(builder, loc, lhs, rhs));
        if (op == "-")  return mlir::Value(circt::comb::SubOp::create(builder, loc, lhs, rhs));
        if (op == "*")  return mlir::Value(circt::comb::MulOp::create(builder, loc, lhs, rhs));
        if (op == "/")  return mlir::Value(circt::comb::DivUOp::create(builder, loc, lhs, rhs));
        if (op == "%")  return mlir::Value(circt::comb::ModUOp::create(builder, loc, lhs, rhs));
        if (op == "&")  return mlir::Value(circt::comb::AndOp::create(builder, loc, lhs, rhs));
        if (op == "|")  return mlir::Value(circt::comb::OrOp::create(builder, loc, lhs, rhs));
        if (op == "^")  return mlir::Value(circt::comb::XorOp::create(builder, loc, lhs, rhs));
        if (op == "<<") return mlir::Value(circt::comb::ShlOp::create(builder, loc, lhs, rhs));
        if (op == ">>") return mlir::Value(circt::comb::ShrUOp::create(builder, loc, lhs, rhs));
        if (op == "&&") return mlir::Value(circt::comb::AndOp::create(builder, loc, lhs, rhs));
        if (op == "||") return mlir::Value(circt::comb::OrOp::create(builder, loc, lhs, rhs));
        if (op == "==") return mlir::Value(circt::comb::ICmpOp::create(builder, loc, circt::comb::ICmpPredicate::eq, lhs, rhs));
        if (op == "!=") return mlir::Value(circt::comb::ICmpOp::create(builder, loc, circt::comb::ICmpPredicate::ne, lhs, rhs));
        if (op == "<")  return mlir::Value(circt::comb::ICmpOp::create(builder, loc, circt::comb::ICmpPredicate::ult, lhs, rhs));
        if (op == "<=") return mlir::Value(circt::comb::ICmpOp::create(builder, loc, circt::comb::ICmpPredicate::ule, lhs, rhs));
        if (op == ">")  return mlir::Value(circt::comb::ICmpOp::create(builder, loc, circt::comb::ICmpPredicate::ugt, lhs, rhs));
        if (op == ">=") return mlir::Value(circt::comb::ICmpOp::create(builder, loc, circt::comb::ICmpPredicate::uge, lhs, rhs));
        notImplemented("binary operator '" + op + "'");
    }

    if (expr.getKind() == ast::ASTNodeKind::UpdateExpr)
    {
        auto &upd = static_cast<ast::UpdateExpr&>(expr);
        auto inner = visit(upd.expr);
        if (!inner.has_value()) return std::any();
        mlir::Value v = std::any_cast<mlir::Value>(inner);
        if (!v) return std::any();

        if (upd.expr->getKind() == ast::ASTNodeKind::IdentExpr)
        {
            std::string nm = static_cast<ast::IdentExpr*>(upd.expr.get())->name;
            if (this->varRegs[this->currentModuleName].count(nm) && this->currentFire)
            {
                mlir::Value one = circt::hw::ConstantOp::create(builder, loc, v.getType(), 1);
                std::string opTxt = upd.op;
                mlir::Value updated = (opTxt == "--")
                    ? mlir::Value(circt::comb::SubOp::create(builder, loc, v, one))
                    : mlir::Value(circt::comb::AddOp::create(builder, loc, v, one));
                writeVar(nm, updated, this->currentFire);
            }
        }
        return v;
    }

    return std::any();
}

std::any LowerVisitor::visitDecl_var(ast::VarDecl &decl)
{
    auto maybeType = getMlirType(decl.typedIdent.type);
    if (!maybeType) return std::any();
    mlir::Location loc = builder.getUnknownLoc();
    for (auto &id : decl.typedIdent.idents)
    {
        std::string name = id;
        if (this->varRegs[this->currentModuleName].count(name)) continue;
        this->variables[this->currentModuleName][name] = maybeType.value();
        mlir::Value zero = circt::hw::ConstantOp::create(builder, loc, maybeType.value(), 0);
        circt::seq::CompRegOp reg = circt::seq::CompRegOp::create(
            builder, loc, zero, this->currentClock, this->currentReset, zero);
        this->varRegs[this->currentModuleName][name] = reg;
        this->varNext[this->currentModuleName][name] = reg.getResult();
    }
    if (decl.initExpr)
    {
        auto rhs = visit(decl.initExpr);
        if (rhs.has_value() && this->currentFire)
        {
            mlir::Value v = std::any_cast<mlir::Value>(rhs);
            if (v && !decl.typedIdent.idents.empty())
                writeVar(decl.typedIdent.idents[0], v, this->currentFire);
        }
    }
    return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// Statements
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitStmt(ast::Stmt &stmt)
{
    return visit(&stmt);
}

std::any LowerVisitor::visitStmt_if(ast::IfStmt &stmt)
{
    mlir::Location loc = builder.getUnknownLoc();
    auto condA = visit(stmt.cond);
    if (!condA.has_value()) return std::any();
    mlir::Value cond = std::any_cast<mlir::Value>(condA);
    if (!cond) return std::any();

    if (auto it = mlir::dyn_cast<mlir::IntegerType>(cond.getType()))
        if (it.getWidth() != 1)
            cond = circt::comb::ICmpOp::create(
                builder, loc, circt::comb::ICmpPredicate::ne, cond,
                circt::hw::ConstantOp::create(builder, loc, cond.getType(), 0));

    mlir::Value outerFire = this->currentFire;
    mlir::Value thenFire = outerFire
        ? mlir::Value(circt::comb::AndOp::create(builder, loc, outerFire, cond))
        : cond;

    if (stmt.thenBranch)
    {
        this->currentFire = thenFire;
        for (auto &s : stmt.thenBranch->statements) visit(s);
        this->currentFire = outerFire;
    }

    mlir::Value notCond = circt::comb::XorOp::create(
        builder, loc, cond,
        circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
    mlir::Value elseFire = outerFire
        ? mlir::Value(circt::comb::AndOp::create(builder, loc, outerFire, notCond))
        : notCond;

    if (stmt.elseBranch)
    {
        this->currentFire = elseFire;
        visit(stmt.elseBranch);
        this->currentFire = outerFire;
    }
    return std::any();
}

std::any LowerVisitor::visitStmt_binary(ast::BinaryStmt &stmt)
{
    mlir::Location loc = builder.getUnknownLoc();
    std::string op = stmt.op;

    auto lhsExpr = stmt.lhs;
    auto rhsExpr = stmt.rhs;

    std::string lhsName;
    bool lhsIsVar = false;
    if (lhsExpr->getKind() == ast::ASTNodeKind::IdentExpr)
    {
        lhsName = static_cast<ast::IdentExpr*>(lhsExpr.get())->name;
        lhsIsVar = true;
    }

    mlir::Type hint;
    if (lhsIsVar && this->variables[this->currentModuleName].count(lhsName))
        hint = this->variables[this->currentModuleName][lhsName];
    mlir::Type prevHint = this->currentExprType;
    this->currentExprType = hint;
    auto rhsAny = visit(rhsExpr);
    this->currentExprType = prevHint;
    mlir::Value rhsVal;
    if (rhsAny.has_value() && rhsAny.type() == typeid(mlir::Value))
        rhsVal = std::any_cast<mlir::Value>(rhsAny);

    mlir::Value fire = this->currentFire
        ? this->currentFire
        : circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);

    if (op == "<-" || op == "->")
    {
        bool rev = (op == "->");
        auto srcExpr = rev ? lhsExpr : rhsExpr;
        auto dstExpr = rev ? rhsExpr : lhsExpr;

        std::string dstName;
        std::string dstInst, dstPort;
        bool dstIsIdent = false, dstIsField = false;
        if (dstExpr->getKind() == ast::ASTNodeKind::IdentExpr)
        {
            dstName = static_cast<ast::IdentExpr*>(dstExpr.get())->name;
            dstIsIdent = true;
        }
        else if (dstExpr->getKind() == ast::ASTNodeKind::FieldExpr)
        {
            dstInst = static_cast<ast::FieldExpr*>(dstExpr.get())->object;
            dstPort = static_cast<ast::FieldExpr*>(dstExpr.get())->field;
            dstIsField = true;
        }

        auto srcAny = visit(srcExpr);
        mlir::Value srcVal;
        if (srcAny.has_value() && srcAny.type() == typeid(mlir::Value))
            srcVal = std::any_cast<mlir::Value>(srcAny);

        if (srcVal)
        {
            if (auto chSrc = mlir::dyn_cast<mlir::TypedValue<circt::esi::ChannelType>>(srcVal))
            {
                mlir::Type inner = chSrc.getType().getInner();
                circt::esi::UnwrapValidReadyOp unwrap = circt::esi::UnwrapValidReadyOp::create(
                    builder, loc, chSrc, fire);
                mlir::Value data  = unwrap.getRawOutput();
                mlir::Value valid = unwrap.getValid();
                mlir::Value cond  = circt::comb::AndOp::create(builder, loc, fire, valid);

                if (dstIsIdent)
                {
                    if (!this->varRegs[this->currentModuleName].count(dstName))
                    {
                        mlir::Value zero = circt::hw::ConstantOp::create(builder, loc, inner, 0);
                        circt::seq::CompRegOp reg = circt::seq::CompRegOp::create(
                            builder, loc, zero, this->currentClock, this->currentReset, zero);
                        this->varRegs[this->currentModuleName][dstName] = reg;
                        this->varNext[this->currentModuleName][dstName] = reg.getResult();
                        this->variables[this->currentModuleName][dstName] = inner;
                    }
                    writeVar(dstName, data, cond);
                    if (this->inStateHeader)
                    {
                        this->currentFire = cond;
                        this->localBindings[dstName] = data;
                    }
                }
                return std::any();
            }
        }

        if (dstIsIdent && this->portDirs[this->currentModuleName].count(dstName) &&
            this->portDirs[this->currentModuleName][dstName] ==
                circt::hw::ModulePort::Direction::Output && srcVal)
        {
            mlir::Type portType;
            for (auto &p : this->currentPorts)
                if (p.name == dstName) { portType = p.type; break; }
            auto chTy = mlir::dyn_cast<circt::esi::ChannelType>(portType);
            if (!chTy) return std::any();
            mlir::Value zero   = circt::hw::ConstantOp::create(builder, loc, chTy.getInner(), 0);
            mlir::Value falseV = circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
            mlir::Value prevData  = this->outputDataNext[this->currentModuleName][dstName];
            mlir::Value prevValid = this->outputValidNext[this->currentModuleName][dstName];
            if (!prevData)  prevData  = zero;
            if (!prevValid) prevValid = falseV;
            mlir::Value coercedSrc = coerce(srcVal, chTy.getInner(), loc);
            mlir::Value newData  = circt::comb::MuxOp::create(builder, loc, fire, coercedSrc, prevData);
            mlir::Value newValid = circt::comb::OrOp::create(builder, loc, fire, prevValid);
            this->outputDataNext[this->currentModuleName][dstName]  = newData;
            this->outputValidNext[this->currentModuleName][dstName] = newValid;
            return std::any();
        }

        if (dstIsField && this->insideInsantiate && srcVal)
        {
            auto instIt = this->instanceFifoRegs.find(dstInst);
            if (instIt != this->instanceFifoRegs.end())
            {
                auto portIt = instIt->second.find(dstPort);
                if (portIt != instIt->second.end())
                {
                    auto [dataReg, validReg] = portIt->second;
                    mlir::Type innerTy = dataReg.getResult().getType();
                    mlir::Value data = coerce(srcVal, innerTy, loc);
                    mlir::Value trueV = circt::hw::ConstantOp::create(
                        builder, loc, builder.getI1Type(), 1);
                    dataReg->setOperand(0, data);
                    dataReg->setOperand(3, data);
                    validReg->setOperand(0, trueV);
                    validReg->setOperand(3, trueV);
                }
            }
        }
        return std::any();
    }

    if (lhsIsVar && this->varRegs[this->currentModuleName].count(lhsName) && rhsVal)
    {
        mlir::Value current = readVar(lhsName);
        mlir::Value newVal;
        if (op == "=" || op == ":=")
        {
            newVal = rhsVal;
        }
        else if (op == "+=")
        {
            newVal = circt::comb::AddOp::create(builder, loc, current, coerce(rhsVal, current.getType(), loc));
        }
        else if (op == "-=")
        {
            newVal = circt::comb::SubOp::create(builder, loc, current, coerce(rhsVal, current.getType(), loc));
        }
        else if (op == "*=")
        {
            newVal = circt::comb::MulOp::create(builder, loc, current, coerce(rhsVal, current.getType(), loc));
        }
        else if (op == "/=")
        {
            newVal = circt::comb::DivUOp::create(builder, loc, current, coerce(rhsVal, current.getType(), loc));
        }
        else if (op == "^=")
        {
            newVal = circt::comb::XorOp::create(builder, loc, current, coerce(rhsVal, current.getType(), loc));
        }
        else if (op == "<<=")
        {
            newVal = circt::comb::ShlOp::create(builder, loc, current, coerce(rhsVal, current.getType(), loc));
        }
        else if (op == ">>=")
        {
            newVal = circt::comb::ShrUOp::create(builder, loc, current, coerce(rhsVal, current.getType(), loc));
        }
        else
        {
            newVal = rhsVal;
        }
        writeVar(lhsName, newVal, fire);
        return std::any();
    }

    return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// Instantiation (top-level)
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitDecl_instantiate(ast::InstantiateDecl &decl)
{
    mlir::Location loc = builder.getUnknownLoc();
    builder.setInsertionPointToEnd(this->topModule.getBody());

    llvm::SmallVector<circt::hw::PortInfo, 2> mainPorts;
    circt::hw::PortInfo clkPort;
    clkPort.name = builder.getStringAttr("clk");
    clkPort.type = circt::seq::ClockType::get(&this->ctx);
    clkPort.dir = circt::hw::ModulePort::Direction::Input;
    mainPorts.push_back(clkPort);

    circt::hw::PortInfo rstPort;
    rstPort.name = builder.getStringAttr("rst");
    rstPort.type = builder.getI1Type();
    rstPort.dir = circt::hw::ModulePort::Direction::Input;
    mainPorts.push_back(rstPort);

    circt::hw::ModulePortInfo mainPortInfo(mainPorts);
    circt::hw::HWModuleOp mainMod = circt::hw::HWModuleOp::create(
        builder, loc, builder.getStringAttr("Main"), mainPortInfo);

    mlir::Block *body = &mainMod.getBody().front();
    for (auto &op : llvm::make_early_inc_range(body->getOperations()))
        if (mlir::isa<circt::hw::OutputOp>(op)) op.erase();
    builder.setInsertionPointToStart(body);
    this->currentClock = body->getArgument(0);
    this->currentReset = body->getArgument(1);

    this->insideInsantiate = true;
    std::string moduleName = "instantiate";
    this->variables[moduleName];
    this->channels[moduleName];
    this->states[moduleName];
    this->portValues[moduleName];
    this->varRegs[moduleName];
    this->varNext[moduleName];
    this->currentModuleName = moduleName;

    std::any result;
    if (decl.body) {
        for (auto &s : decl.body->statements) {
            result = visit(s);
        }
    }

    circt::hw::OutputOp::create(builder, loc, mlir::ValueRange{});
    builder.setInsertionPointToEnd(this->topModule.getBody());

    this->insideInsantiate = false;
    return result;
}

std::any LowerVisitor::visitInst_module(ast::InstModuleStmt &stmt)
{
    std::string moduleName = stmt.callExpr->funcName;
    std::string varName = stmt.varName;

    if (!this->modules.contains(moduleName))
    {
        std::cerr << "Error: module '" << moduleName << "' not found for instantiation!\n";
        exit(1);
    }
    circt::hw::HWModuleOp mod = this->modules[moduleName];
    llvm::SmallVector<circt::hw::PortInfo, 4> portInfo = mod.getPortList();
    std::vector<mlir::Value> instanceOperands;

    for (auto port : portInfo)
    {
        if (port.dir == circt::hw::ModulePort::Direction::Output) continue;

        mlir::Type portType = port.type;
        if (auto channelType = mlir::dyn_cast<circt::esi::ChannelType>(portType))
        {
            mlir::Type innerType = channelType.getInner();
            mlir::Value dummyData  = circt::hw::ConstantOp::create(builder, builder.getUnknownLoc(), innerType, 0);
            mlir::Value dummyValid = circt::hw::ConstantOp::create(builder, builder.getUnknownLoc(), builder.getI1Type(), 0);
            circt::seq::CompRegOp channel_data  = circt::seq::CompRegOp::create(builder, builder.getUnknownLoc(), dummyData,  this->currentClock, this->currentReset, dummyData);
            circt::seq::CompRegOp channel_valid = circt::seq::CompRegOp::create(builder, builder.getUnknownLoc(), dummyValid, this->currentClock, this->currentReset, dummyValid);
            this->instanceFifoRegs[varName][port.name.str()] = {channel_data, channel_valid};
            circt::esi::WrapValidReadyOp vr = circt::esi::WrapValidReadyOp::create(
                builder, builder.getUnknownLoc(), portType, builder.getI1Type(), channel_data, channel_valid);
            mlir::TypedValue<circt::esi::ChannelType> ch_in = vr.getChanOutput();
            circt::esi::FIFOOp fifo = circt::esi::FIFOOp::create(builder, builder.getUnknownLoc(),
                                                                 port.type, this->currentClock, this->currentReset, ch_in, 5);
            instanceOperands.push_back(fifo.getResult());
        }
        else if (mlir::isa<circt::seq::ClockType>(portType))
        {
            instanceOperands.push_back(this->currentClock);
        }
        else if (mlir::isa<mlir::IntegerType>(portType) &&
                 mlir::cast<mlir::IntegerType>(portType).getWidth() == 1)
        {
            instanceOperands.push_back(this->currentReset);
        }
        else
        {
            std::cerr << "Error: unsupported port type for port '" << port.name.getValue().str() << "' in module '" << moduleName << "'!\n";
            exit(1);
        }
    }
    circt::hw::InstanceOp instance = circt::hw::InstanceOp::create(
        builder, builder.getUnknownLoc(), mod, varName, instanceOperands);
    if (this->instances.contains(varName))
    {
        std::cerr << "Error: instance '" << varName << "' is already declared!\n";
        exit(1);
    }
    this->instances[varName] = instance;
    return std::any();
}

std::any LowerVisitor::visitIdent_field(ast::FieldExpr &expr)
{
    if (!this->insideInsantiate)
    {
        return std::any();
    }
    auto inst = expr.object;
    auto port = expr.field;
    if (this->instances.count(inst))
    {
        circt::hw::InstanceOp op = this->instances[inst];
        auto plist = op.getPortList();
        unsigned outIdx = 0;
        for (auto &p : plist)
        {
            if (p.dir == circt::hw::ModulePort::Direction::Output && p.name == port)
                return mlir::Value(op.getResult(outIdx));
            if (p.dir == circt::hw::ModulePort::Direction::Output) ++outIdx;
        }
    }
    return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// Dummy / Unimplemented declarations and statements
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitBlockStmt(ast::BlockStmt &stmt)
{
    for (auto &s : stmt.statements)
        visit(s);
    return std::any();
}

std::any LowerVisitor::visitForStmt(ast::ForStmt &)
{
    notImplemented("for loops");
}

std::any LowerVisitor::visitSwitchStmt(ast::SwitchStmt &)
{
    notImplemented("switch statements");
}

std::any LowerVisitor::visitReturnStmt(ast::ReturnStmt &)
{
    notImplemented("return statements");
}
