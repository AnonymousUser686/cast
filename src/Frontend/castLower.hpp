#pragma once
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/SV/SVOps.h>
#include <circt/Dialect/FSM/FSMOps.h>
#include "circt/Dialect/FSM/FSMDialect.h"
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/ESI/ESIDialect.h>
#include <circt/Dialect/ESI/ESIOps.h>
#include <circt/Dialect/Seq/SeqDialect.h>
#include <circt/Dialect/Seq/SeqOps.h>
#include <circt/Dialect/Seq/SeqTypes.h>
#include <unordered_map>
#include <string>
#include <any>
#include <optional>
#include <map>
#include "CastAST.hpp"

class LowerVisitor
{
public:
    mlir::ModuleOp topModule;

    LowerVisitor() : builder(&ctx)
    {
        mlir::Location loc = mlir::UnknownLoc::get(&ctx);
        topModule = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(topModule.getBody());
        this->ctx.getOrLoadDialect<circt::hw::HWDialect>();
        this->ctx.getOrLoadDialect<circt::fsm::FSMDialect>();
        this->ctx.getOrLoadDialect<circt::seq::SeqDialect>();
        this->ctx.getOrLoadDialect<circt::sv::SVDialect>();
        this->ctx.getOrLoadDialect<circt::comb::CombDialect>();
        this->ctx.getOrLoadDialect<circt::esi::ESIDialect>();
        this->currentClock = nullptr;
    }

    std::any visit(ast::ASTNode *node);
    std::any visit(const std::shared_ptr<ast::ASTNode> &node);

    std::any visitDecl_enum(ast::EnumDecl &decl);
    std::any visitDecl_machine(ast::MachineDecl &decl);
    std::any visitDecl_states(ast::StatesBlock &block);
    std::any visitDecl_state(ast::StateDecl &decl);
    std::any visitDecl_interface(ast::InterfaceBlock &block);
    std::any visitDecl_shared(ast::SharedBlock &block);
    std::any visitDecl_instantiate(ast::InstantiateDecl &decl);
    std::any visitInst_module(ast::InstModuleStmt &stmt);
    std::any visitIdent_field(ast::FieldExpr &expr);
    std::any visitIndexExpr(ast::IndexExpr &expr);
    std::any visitStmt_binary(ast::BinaryStmt &stmt);
    std::any visitIdent(ast::IdentExpr &expr);
    std::any visitNumber_literal(ast::NumberLiteral &expr);
    std::any visitStmt(ast::Stmt &stmt);
    std::any visitExpr(ast::Expr &expr);
    std::any visitStmt_if(ast::IfStmt &stmt);
    std::any visitStmt_nextstate(ast::GotoStmt &stmt);
    std::any visitExpr_func_call(ast::CallExpr &expr);
    std::any visitDecl_var(ast::VarDecl &decl);

    // Unimplemented but parsed statements
    std::any visitBlockStmt(ast::BlockStmt &stmt);
    std::any visitForStmt(ast::ForStmt &stmt);
    std::any visitSwitchStmt(ast::SwitchStmt &stmt);
    std::any visitReturnStmt(ast::ReturnStmt &stmt);

    std::optional<mlir::Type> getMlirType(const ast::Type &type);
    std::optional<int64_t> evalConst(ast::Expr *expr, const std::map<std::string, int64_t> &consts);

private:
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder;
    llvm::SmallVector<circt::hw::PortInfo, 4> currentPorts;
    std::string currentModuleName;

    // Type info per module
    std::unordered_map<std::string, std::unordered_map<std::string, mlir::Type>> variables;
    std::unordered_map<std::string, mlir::Type> channels;
    std::unordered_map<std::string, mlir::Type> states;
    std::unordered_map<std::string, circt::hw::HWModuleOp> modules;
    std::unordered_map<std::string, circt::hw::InstanceOp> instances;
    // FIFO feed registers created in visitInst_module, keyed by [instanceName][portName].
    // Patched when the instantiate block contains `m.port <- value`.
    std::unordered_map<std::string,
        std::unordered_map<std::string,
            std::pair<circt::seq::CompRegOp, circt::seq::CompRegOp>>> instanceFifoRegs;
    mlir::Value currentClock;

    // Array storage: arrays lower to one register per element, stored in
    // varRegs under mangled keys ("a[0]", "m[1][2]"). arrayInfo[mod][name]
    // records the dimension sizes and element type of each declared array.
    struct ArrayInfo {
        std::vector<int64_t> dims;
        mlir::Type elemType;
    };
    std::unordered_map<std::string, std::unordered_map<std::string, ArrayInfo>> arrayInfo;

    // Live mlir::Value bindings per module
    // varRegs[mod][name] : the CompRegOp backing a shared variable
    std::unordered_map<std::string, std::unordered_map<std::string, circt::seq::CompRegOp>> varRegs;
    // varNext[mod][name] : the running mux chain that becomes the reg's input
    //                     Each conditional update appends a `mux(cond, newVal, prev)` layer.
    std::unordered_map<std::string, std::unordered_map<std::string, mlir::Value>> varNext;
    // portValues[mod][name] : the block-argument mlir::Value for an interface port
    std::unordered_map<std::string, std::unordered_map<std::string, mlir::Value>> portValues;
    // portDirs[mod][name] : direction of each port (input/output)
    std::unordered_map<std::string, std::unordered_map<std::string, circt::hw::ModulePort::Direction>> portDirs;

    // State machine, per module
    // stateRegs[mod] : the CompRegOp holding the active state ID
    std::unordered_map<std::string, circt::seq::CompRegOp> stateRegs;
    // stateIds[mod][name] : the integer encoding of each state
    std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> stateIds;
    // stateActive[mod][name] : combinational i1 — true when this state is current
    std::unordered_map<std::string, std::unordered_map<std::string, mlir::Value>> stateActive;
    // stateNext[mod] : the chain of goto-driven mux layers feeding the state register
    std::unordered_map<std::string, mlir::Value> stateNext;

    // Output channel drivers, per module
    // outputDataNext / outputValidNext : chains feeding wrap_valid_ready when the module finishes
    std::unordered_map<std::string, std::unordered_map<std::string, mlir::Value>> outputDataNext;
    std::unordered_map<std::string, std::unordered_map<std::string, mlir::Value>> outputValidNext;

    // Enum value table — name → (encoding, width)
    std::unordered_map<std::string, std::pair<uint64_t, unsigned>> enumValues;

    // Per-statement context (mutable while walking a state body)
    mlir::Value currentFire;        // i1: when high, the current statement should "fire"
    std::string currentStateName;   // name of the state currently being lowered
    bool inStateBody = false;       // true while inside a decl_state body
    bool inStateHeader = false;     // true while visiting state header receives
    mlir::Type currentExprType;     // hint for typing literals from context
    mlir::Value currentReset;       // i1: synchronous reset signal (0 in single-clock modules)
    // Within a state body, header receives bind their variable here so reads
    // see the just-received data rather than the previous register value.
    std::unordered_map<std::string, mlir::Value> localBindings;

    // Compile-time values of enclosing for-loop variables. Nested loops merge
    // into this map so inner bounds/indices may reference outer loop vars.
    std::map<std::string, int64_t> loopConstBindings;
    // > 0 while lowering a for-loop body. Inside loops, readVar returns the
    // pending (already-written-this-cycle) value so unrolled iterations see
    // each other's writes; outside loops reads keep the language's
    // registers-commit-at-cycle-end semantics.
    int forLoopDepth = 0;

    bool insideInsantiate = false;

    // Helpers
    mlir::Value readVar(const std::string &name);
    void writeVar(const std::string &name, mlir::Value newVal, mlir::Value cond);
    mlir::Value coerce(mlir::Value v, mlir::Type t, mlir::Location loc);
    mlir::Value applyCompound(const std::string &op, mlir::Value current,
                              mlir::Value rhs, mlir::Location loc);
    void declareArray(const std::string &name, const std::vector<int64_t> &dims,
                      mlir::Type elemType);
    // Resolved array access: element keys the access can hit, with the
    // dynamic-index comparison condition for each (null when fully constant).
    struct ArrayAccess {
        std::vector<std::pair<std::string, mlir::Value>> elems;
        mlir::Type elemType;
    };
    std::optional<ArrayAccess> resolveArrayAccess(ast::Expr *expr);
    void writeArray(ast::Expr *lhs, const std::string &op, mlir::Value rhsVal,
                    mlir::Value fire);
};