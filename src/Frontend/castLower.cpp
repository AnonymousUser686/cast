#include "castLower.hpp"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/ESI/ESIOps.h>
#include <circt/Dialect/ESI/ESITypes.h>
#include <circt/Dialect/FSM/FSMOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HW/HWTypes.h>
#include <circt/Dialect/SV/SVAttributes.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <circt/Dialect/SV/SVOps.h>
#include <circt/Dialect/SV/SVTypes.h>
#include <circt/Dialect/Seq/SeqDialect.h>
#include <circt/Dialect/Seq/SeqOps.h>
#include <iostream>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// Error helpers
// ─────────────────────────────────────────────────────────────────────────────

[[noreturn]] static void notImplemented(const std::string &construct) {
  std::cerr << "Error: '" << construct
            << "' is not yet supported by the cast compiler.\n";
  exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// AST Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visit(ast::ASTNode *node) {
  if (!node)
    return std::any();
  switch (node->getKind()) {
  case ast::ASTNodeKind::Program:
    for (auto &decl : static_cast<ast::Program *>(node)->decls) {
      visit(decl.get());
    }
    return std::any();
  case ast::ASTNodeKind::EnumDecl:
    return visitDecl_enum(*static_cast<ast::EnumDecl *>(node));
  case ast::ASTNodeKind::MachineDecl:
    return visitDecl_machine(*static_cast<ast::MachineDecl *>(node));
  case ast::ASTNodeKind::InterfaceBlock:
    return visitDecl_interface(*static_cast<ast::InterfaceBlock *>(node));
  case ast::ASTNodeKind::SharedBlock:
    return visitDecl_shared(*static_cast<ast::SharedBlock *>(node));
  case ast::ASTNodeKind::StatesBlock:
    return visitDecl_states(*static_cast<ast::StatesBlock *>(node));
  case ast::ASTNodeKind::StateDecl:
    return visitDecl_state(*static_cast<ast::StateDecl *>(node));
  case ast::ASTNodeKind::InstantiateDecl:
    return visitDecl_instantiate(*static_cast<ast::InstantiateDecl *>(node));
  case ast::ASTNodeKind::VarDecl:
    return visitDecl_var(*static_cast<ast::VarDecl *>(node));

  // Statements
  case ast::ASTNodeKind::BlockStmt:
    return visitBlockStmt(*static_cast<ast::BlockStmt *>(node));
  case ast::ASTNodeKind::VarDeclStmt:
    return visitDecl_var(*static_cast<ast::VarDeclStmt *>(node)->varDecl);
  case ast::ASTNodeKind::InstModuleStmt:
    return visitInst_module(*static_cast<ast::InstModuleStmt *>(node));
  case ast::ASTNodeKind::IfStmt:
    return visitStmt_if(*static_cast<ast::IfStmt *>(node));
  case ast::ASTNodeKind::BinaryStmt:
    return visitStmt_binary(*static_cast<ast::BinaryStmt *>(node));
  case ast::ASTNodeKind::GotoStmt:
    return visitStmt_nextstate(*static_cast<ast::GotoStmt *>(node));
  case ast::ASTNodeKind::ExprStmt:
    return visit(static_cast<ast::ExprStmt *>(node)->expr.get());
  case ast::ASTNodeKind::ForStmt:
    return visitForStmt(*static_cast<ast::ForStmt *>(node));
  case ast::ASTNodeKind::SwitchStmt:
    return visitSwitchStmt(*static_cast<ast::SwitchStmt *>(node));
  case ast::ASTNodeKind::ReturnStmt:
    return visitReturnStmt(*static_cast<ast::ReturnStmt *>(node));

  // Expressions
  case ast::ASTNodeKind::IdentExpr:
    return visitIdent(*static_cast<ast::IdentExpr *>(node));
  case ast::ASTNodeKind::FieldExpr:
    return visitIdent_field(*static_cast<ast::FieldExpr *>(node));
  case ast::ASTNodeKind::IndexExpr:
    return visitIndexExpr(*static_cast<ast::IndexExpr *>(node));
  case ast::ASTNodeKind::NumberLiteral:
    return visitNumber_literal(*static_cast<ast::NumberLiteral *>(node));
  case ast::ASTNodeKind::StringLiteral:
    return std::any();
  case ast::ASTNodeKind::NilLiteral:
    return std::any();
  case ast::ASTNodeKind::UnaryExpr:
  case ast::ASTNodeKind::BinaryExpr:
  case ast::ASTNodeKind::UpdateExpr:
    return visitExpr(*static_cast<ast::Expr *>(node));
  case ast::ASTNodeKind::CallExpr:
    return visitExpr_func_call(*static_cast<ast::CallExpr *>(node));

  default:
    return std::any();
  }
}

std::any LowerVisitor::visit(const std::shared_ptr<ast::ASTNode> &node) {
  return visit(node.get());
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

mlir::Value LowerVisitor::coerce(mlir::Value v, mlir::Type t,
                                 mlir::Location loc) {
  if (!v || !t)
    return v;
  auto srcInt = mlir::dyn_cast<mlir::IntegerType>(v.getType());
  auto dstInt = mlir::dyn_cast<mlir::IntegerType>(t);
  if (!srcInt || !dstInt)
    return v;

  if (!srcInt.isSignless()) {
    mlir::Type signless = mlir::IntegerType::get(&this->ctx, srcInt.getWidth());
    v = mlir::UnrealizedConversionCastOp::create(builder, loc, signless,
                                                 mlir::ValueRange{v})
            .getResult(0);
    srcInt = mlir::cast<mlir::IntegerType>(v.getType());
  }
  mlir::Type targetSignless =
      dstInt.isSignless()
          ? t
          : mlir::IntegerType::get(&this->ctx, dstInt.getWidth());

  if (v.getType() == targetSignless)
    return v;
  if (srcInt.getWidth() == dstInt.getWidth())
    return v;
  if (srcInt.getWidth() < dstInt.getWidth()) {
    unsigned padW = dstInt.getWidth() - srcInt.getWidth();
    mlir::Value zero = circt::hw::ConstantOp::create(
        builder, loc, builder.getIntegerType(padW), 0);
    return circt::comb::ConcatOp::create(builder, loc,
                                         mlir::ValueRange{zero, v});
  }
  return circt::comb::ExtractOp::create(builder, loc, targetSignless, v, 0);
}

mlir::Value LowerVisitor::readVar(const std::string &name) {
  auto &mvars = this->varRegs[this->currentModuleName];
  auto it = mvars.find(name);
  if (it == mvars.end())
    return nullptr;
  // Inside an unrolled for-loop body, reads observe writes made earlier in
  // the same cycle (software order), so accumulations like `sum += a[i]`
  // work when unrolled. Outside loops, reads keep the documented semantics:
  // they return the pre-update register value.
  if (this->forLoopDepth > 0) {
    auto &mnext = this->varNext[this->currentModuleName];
    auto nIt = mnext.find(name);
    if (nIt != mnext.end() && nIt->second)
      return nIt->second;
  }
  return it->second.getResult();
}

void LowerVisitor::writeVar(const std::string &name, mlir::Value newVal,
                            mlir::Value cond) {
  auto &mvars = this->varRegs[this->currentModuleName];
  auto it = mvars.find(name);
  if (it == mvars.end())
    return;
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Value prev = this->varNext[this->currentModuleName][name];
  mlir::Value coerced = coerce(newVal, prev.getType(), loc);
  mlir::Value next =
      circt::comb::MuxOp::create(builder, loc, cond, coerced, prev);
  this->varNext[this->currentModuleName][name] = next;
}

// Shared implementation of the compound-assignment operators for both scalar
// variables and array elements. `current` is the present value of the target;
// returns the value to store.
mlir::Value LowerVisitor::applyCompound(const std::string &op,
                                        mlir::Value current, mlir::Value rhs,
                                        mlir::Location loc) {
  if (op == "=" || op == ":=")
    return rhs;
  mlir::Value r = coerce(rhs, current.getType(), loc);
  if (op == "+=")
    return circt::comb::AddOp::create(builder, loc, current, r);
  if (op == "-=")
    return circt::comb::SubOp::create(builder, loc, current, r);
  if (op == "*=")
    return circt::comb::MulOp::create(builder, loc, current, r);
  if (op == "/=")
    return circt::comb::DivUOp::create(builder, loc, current, r);
  if (op == "^=")
    return circt::comb::XorOp::create(builder, loc, current, r);
  if (op == "<<=")
    return circt::comb::ShlOp::create(builder, loc, current, r);
  if (op == ">>=")
    return circt::comb::ShrUOp::create(builder, loc, current, r);
  return rhs;
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrays
//
// An `[N]T` array lowers to N element registers (a register file). Constant
// indices resolve to a single element at compile time; dynamic indices build
// a mux tree on reads and per-element write-enable conditions on writes.
// ─────────────────────────────────────────────────────────────────────────────

// Peels a[i][j] → ("a", {i, j}). Returns false when the base is not a plain
// identifier.
static bool peelIndexChain(ast::Expr *expr, std::string &name,
                           std::vector<ast::Expr *> &indices) {
  while (expr && expr->getKind() == ast::ASTNodeKind::IndexExpr) {
    auto *ix = static_cast<ast::IndexExpr *>(expr);
    indices.insert(indices.begin(), ix->index.get());
    expr = ix->base.get();
  }
  if (expr && expr->getKind() == ast::ASTNodeKind::IdentExpr) {
    name = static_cast<ast::IdentExpr *>(expr)->name;
    return true;
  }
  return false;
}

static std::string elemKey(const std::string &name,
                           const std::vector<int64_t> &idx) {
  std::string key = name;
  for (int64_t i : idx)
    key += "[" + std::to_string(i) + "]";
  return key;
}

void LowerVisitor::declareArray(const std::string &name,
                                const std::vector<int64_t> &dims,
                                mlir::Type elemType) {
  mlir::Location loc = builder.getUnknownLoc();
  int64_t total = 1;
  for (int64_t d : dims)
    total *= d;
  if (total > 65536) {
    std::cerr << "Error: array '" << name << "' has " << total
              << " elements; the register-per-element lowering supports at "
                 "most 65536.\n";
    exit(1);
  }

  this->arrayInfo[this->currentModuleName][name] = {dims, elemType};

  std::vector<int64_t> idx(dims.size(), 0);
  for (int64_t flat = 0; flat < total; ++flat) {
    std::string key = elemKey(name, idx);
    this->variables[this->currentModuleName][key] = elemType;
    mlir::Value zero = circt::hw::ConstantOp::create(builder, loc, elemType, 0);
    circt::seq::CompRegOp reg = circt::seq::CompRegOp::create(
        builder, loc, zero, this->currentClock, this->currentReset, zero);
    this->varRegs[this->currentModuleName][key] = reg;
    this->varNext[this->currentModuleName][key] = reg.getResult();
    // Advance the multi-dimensional index (odometer order, last dim fastest).
    for (int d = (int)dims.size() - 1; d >= 0; --d) {
      if (++idx[d] < dims[d])
        break;
      idx[d] = 0;
    }
  }
}

std::optional<LowerVisitor::ArrayAccess>
LowerVisitor::resolveArrayAccess(ast::Expr *expr) {
  std::string name;
  std::vector<ast::Expr *> indices;
  if (!peelIndexChain(expr, name, indices))
    return std::nullopt;

  auto &arrays = this->arrayInfo[this->currentModuleName];
  auto arrIt = arrays.find(name);
  if (arrIt == arrays.end()) {
    std::cerr << "Error: '" << name
              << "' is indexed with [] but is not a declared array.\n";
    exit(1);
  }
  const ArrayInfo &info = arrIt->second;
  const std::vector<int64_t> &dims = info.dims;
  if (indices.size() != dims.size()) {
    std::cerr << "Error: array '" << name << "' has " << dims.size()
              << " dimension(s) but is accessed with " << indices.size()
              << " index/indices.\n";
    exit(1);
  }

  mlir::Location loc = builder.getUnknownLoc();

  // Resolve each index: compile-time constant when possible, otherwise a
  // hardware value to compare against at runtime.
  std::vector<std::optional<int64_t>> constIdx(dims.size());
  std::vector<mlir::Value> dynIdx(dims.size());
  for (size_t d = 0; d < dims.size(); ++d) {
    auto c = evalConst(indices[d], this->loopConstBindings);
    if (c) {
      if (*c < 0 || *c >= dims[d]) {
        std::cerr << "Error: index " << *c << " is out of bounds for array '"
                  << name << "' dimension " << d << " (size " << dims[d]
                  << ").\n";
        exit(1);
      }
      constIdx[d] = *c;
      continue;
    }
    auto any = visit(indices[d]);
    if (!any.has_value() || any.type() != typeid(mlir::Value)) {
      std::cerr << "Error: cannot evaluate index expression for array '"
                << name << "'.\n";
      exit(1);
    }
    mlir::Value v = std::any_cast<mlir::Value>(any);
    if (!v || !mlir::isa<mlir::IntegerType>(v.getType())) {
      std::cerr << "Error: index expression for array '" << name
                << "' is not an integer value.\n";
      exit(1);
    }
    dynIdx[d] = v;
  }

  // Enumerate the elements this access can reach. Constant dimensions pin
  // their coordinate; dynamic dimensions contribute an equality comparison.
  ArrayAccess access;
  access.elemType = info.elemType;
  int64_t total = 1;
  for (int64_t d : dims)
    total *= d;
  std::vector<int64_t> idx(dims.size(), 0);
  for (int64_t flat = 0; flat < total; ++flat) {
    bool reachable = true;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (constIdx[d] && *constIdx[d] != idx[d]) {
        reachable = false;
        break;
      }
      if (dynIdx[d]) {
        unsigned w =
            mlir::cast<mlir::IntegerType>(dynIdx[d].getType()).getWidth();
        // Coordinates the index register cannot even represent are dead.
        if (w < 64 && idx[d] >= (int64_t{1} << w)) {
          reachable = false;
          break;
        }
      }
    }
    if (reachable) {
      mlir::Value cond; // null = unconditional (all dims constant)
      for (size_t d = 0; d < dims.size(); ++d) {
        if (!dynIdx[d])
          continue;
        mlir::Value coord = circt::hw::ConstantOp::create(
            builder, loc, dynIdx[d].getType(), idx[d]);
        mlir::Value eq = circt::comb::ICmpOp::create(
            builder, loc, circt::comb::ICmpPredicate::eq, dynIdx[d], coord);
        cond = cond ? mlir::Value(
                          circt::comb::AndOp::create(builder, loc, cond, eq))
                    : eq;
      }
      access.elems.push_back({elemKey(name, idx), cond});
    }
    for (int d = (int)dims.size() - 1; d >= 0; --d) {
      if (++idx[d] < dims[d])
        break;
      idx[d] = 0;
    }
  }
  return access;
}

std::any LowerVisitor::visitIndexExpr(ast::IndexExpr &expr) {
  auto access = resolveArrayAccess(&expr);
  if (!access)
    return std::any();
  mlir::Location loc = builder.getUnknownLoc();

  // Fully constant access → read the one element directly.
  if (access->elems.size() == 1 && !access->elems[0].second)
    return mlir::Value(readVar(access->elems[0].first));

  // Dynamic access → mux tree over every reachable element. An out-of-range
  // index value reads as 0.
  mlir::Value result =
      circt::hw::ConstantOp::create(builder, loc, access->elemType, 0);
  for (auto &[key, cond] : access->elems) {
    mlir::Value elemVal = readVar(key);
    if (!elemVal)
      continue;
    result = cond ? mlir::Value(circt::comb::MuxOp::create(builder, loc, cond,
                                                           elemVal, result))
                  : elemVal;
  }
  return result;
}

void LowerVisitor::writeArray(ast::Expr *lhs, const std::string &op,
                              mlir::Value rhsVal, mlir::Value fire) {
  auto access = resolveArrayAccess(lhs);
  if (!access)
    return;
  mlir::Location loc = builder.getUnknownLoc();
  for (auto &[key, cond] : access->elems) {
    mlir::Value enable =
        cond ? mlir::Value(circt::comb::AndOp::create(builder, loc, fire, cond))
             : fire;
    mlir::Value current = readVar(key);
    if (!current)
      continue;
    mlir::Value newVal = applyCompound(op, current, rhsVal, loc);
    writeVar(key, newVal, enable);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level declarations
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitDecl_enum(ast::EnumDecl &decl) {
  unsigned width = 8;
  for (size_t i = 0; i < decl.members.size(); ++i) {
    std::string name = decl.members[i];
    this->enumValues[name] = {static_cast<uint64_t>(i), width};
  }
  return std::any();
}

std::any LowerVisitor::visitDecl_machine(ast::MachineDecl &decl) {
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
  this->arrayInfo[moduleName];
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
  circt::hw::HWModuleOp hwModule =
      circt::hw::HWModuleOp::create(b, loc, modNameAttr, portInfo);
  this->modules[moduleName] = hwModule;

  mlir::Block *body = &hwModule.getBody().front();
  for (auto &op : llvm::make_early_inc_range(body->getOperations()))
    if (mlir::isa<circt::hw::OutputOp>(op))
      op.erase();

  this->currentClock = body->getArgument(0);
  unsigned argIdx = 0;
  for (auto &p : this->currentPorts) {
    std::string pname = p.name.str();
    this->portDirs[moduleName][pname] = p.dir;
    if (p.dir == circt::hw::ModulePort::Direction::Input) {
      this->portValues[moduleName][pname] = body->getArgument(argIdx++);
    }
  }
  b.setInsertionPointToStart(body);
  this->currentReset = body->getArgument(1);

  if (decl.sharedBlock)
    visit(decl.sharedBlock);
  if (decl.statesBlock)
    visit(decl.statesBlock);

  for (auto &kv : this->varRegs[moduleName]) {
    const std::string &name = kv.first;
    circt::seq::CompRegOp reg = kv.second;
    mlir::Value next = this->varNext[moduleName][name];
    if (next)
      reg->setOperand(0, next);
  }
  if (this->stateRegs.count(moduleName)) {
    circt::seq::CompRegOp sreg = this->stateRegs[moduleName];
    mlir::Value next = this->stateNext[moduleName];
    if (next)
      sreg->setOperand(0, next);
  }
  llvm::SmallVector<mlir::Value, 4> outputOperands;
  for (auto &p : this->currentPorts) {
    if (p.dir != circt::hw::ModulePort::Direction::Output)
      continue;
    std::string pname = p.name.str();
    mlir::Value data = this->outputDataNext[moduleName][pname];
    mlir::Value valid = this->outputValidNext[moduleName][pname];
    auto chTy = mlir::dyn_cast<circt::esi::ChannelType>(p.type);
    if (!data && chTy)
      data = circt::hw::ConstantOp::create(b, loc, chTy.getInner(), 0);
    if (!valid)
      valid = circt::hw::ConstantOp::create(b, loc, b.getI1Type(), 0);
    if (chTy) {
      circt::esi::WrapValidReadyOp wrap = circt::esi::WrapValidReadyOp::create(
          b, loc, p.type, b.getI1Type(), data, valid);
      outputOperands.push_back(wrap.getResult(0));
    }
  }
  circt::hw::OutputOp::create(b, loc, outputOperands);

  b.setInsertionPointToEnd(this->topModule.getBody());

  return std::any();
}

std::any LowerVisitor::visitDecl_interface(ast::InterfaceBlock &block) {
  for (auto &decl : block.ioDecls) {
    for (auto &id : decl.idents) {
      circt::hw::PortInfo port;
      port.name = builder.getStringAttr(id);
      auto maybeType = getMlirType(decl.typedIdent.type);
      mlir::Type inner = maybeType.value_or(builder.getI8Type());
      port.type = circt::esi::ChannelType::get(&this->ctx, inner);
      this->channels[id] = port.type;
      if (decl.direction == "input") {
        port.dir = circt::hw::ModulePort::Direction::Input;
      } else if (decl.direction == "output") {
        port.dir = circt::hw::ModulePort::Direction::Output;
      } else {
        std::cerr << "Error: machine port must be 'input' or 'output'.\n";
        exit(1);
      }
      this->currentPorts.push_back(port);
    }
  }
  return std::any();
}

std::any LowerVisitor::visitDecl_shared(ast::SharedBlock &block) {
  mlir::Location loc = builder.getUnknownLoc();
  for (auto &v : block.varDecls) {
    // Array declaration: peel [N] layers to the element type, then create
    // one register per element.
    if (v->typedIdent.type.kind == ast::TypeKind::ARRAY) {
      std::vector<int64_t> dims;
      const ast::Type *t = &v->typedIdent.type;
      while (t->kind == ast::TypeKind::ARRAY && t->elementType) {
        dims.push_back(t->size);
        t = t->elementType.get();
      }
      auto elemType = getMlirType(*t);
      if (!elemType) {
        std::cerr << "Error: array element type '" << t->name
                  << "' is not supported.\n";
        exit(1);
      }
      if (v->initExpr) {
        std::cerr << "Error: array initializers are not supported; assign "
                     "elements in a state instead.\n";
        exit(1);
      }
      for (auto &id : v->typedIdent.idents)
        declareArray(id, dims, elemType.value());
      continue;
    }

    auto maybeType = getMlirType(v->typedIdent.type);
    if (!maybeType)
      continue;

    // A constant initializer becomes the register's reset value.
    int64_t initVal = 0;
    if (v->initExpr) {
      auto c = evalConst(v->initExpr.get(), {});
      if (!c) {
        std::cerr << "Error: shared variable initializer must be a "
                     "compile-time constant.\n";
        exit(1);
      }
      initVal = *c;
    }

    for (auto &id : v->typedIdent.idents) {
      std::string name = id;
      this->variables[this->currentModuleName][name] = maybeType.value();
      mlir::Value init = circt::hw::ConstantOp::create(
          builder, loc, maybeType.value(), initVal);
      circt::seq::CompRegOp reg = circt::seq::CompRegOp::create(
          builder, loc, init, this->currentClock, this->currentReset, init);
      this->varRegs[this->currentModuleName][name] = reg;
      this->varNext[this->currentModuleName][name] = reg.getResult();
    }
  }
  return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// State machine
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitDecl_states(ast::StatesBlock &block) {
  mlir::Location loc = builder.getUnknownLoc();
  auto &stateDecls = block.stateDecls;
  if (stateDecls.empty())
    return std::any();

  unsigned needed = 1;
  while ((1u << needed) < stateDecls.size())
    ++needed;
  if (needed < 1)
    needed = 1;
  mlir::Type stateTy = builder.getIntegerType(needed);

  uint32_t id = 0;
  for (auto &sd : stateDecls) {
    std::string name = sd->name;
    this->stateIds[this->currentModuleName][name] = id;
    ++id;
  }

  mlir::Value resetVal =
      circt::hw::ConstantOp::create(builder, loc, stateTy, 0);
  circt::seq::CompRegOp stateReg = circt::seq::CompRegOp::create(
      builder, loc, resetVal, this->currentClock, this->currentReset, resetVal);
  this->stateRegs[this->currentModuleName] = stateReg;
  this->stateNext[this->currentModuleName] = stateReg.getResult();

  for (auto &kv : this->stateIds[this->currentModuleName]) {
    mlir::Value sId =
        circt::hw::ConstantOp::create(builder, loc, stateTy, kv.second);
    mlir::Value eq = circt::comb::ICmpOp::create(builder, loc,
                                                 circt::comb::ICmpPredicate::eq,
                                                 stateReg.getResult(), sId);
    this->stateActive[this->currentModuleName][kv.first] = eq;
  }

  for (auto &sd : stateDecls)
    visit(sd);

  return std::any();
}

std::any LowerVisitor::visitDecl_state(ast::StateDecl &decl) {
  std::string name = decl.name;
  mlir::Value active = this->stateActive[this->currentModuleName][name];
  if (!active)
    return std::any();

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

std::any LowerVisitor::visitStmt_nextstate(ast::GotoStmt &stmt) {
  std::string target = stmt.targetState;
  if (target.empty())
    return std::any();

  auto &ids = this->stateIds[this->currentModuleName];
  auto it = ids.find(target);
  if (it == ids.end())
    return std::any();

  mlir::Location loc = builder.getUnknownLoc();
  circt::seq::CompRegOp sreg = this->stateRegs[this->currentModuleName];
  mlir::Type stateTy = sreg.getResult().getType();
  mlir::Value targetId =
      circt::hw::ConstantOp::create(builder, loc, stateTy, it->second);

  mlir::Value prev = this->stateNext[this->currentModuleName];
  mlir::Value next = circt::comb::MuxOp::create(builder, loc, this->currentFire,
                                                targetId, prev);
  this->stateNext[this->currentModuleName] = next;
  return std::any();
}

std::any LowerVisitor::visitExpr_func_call(ast::CallExpr &expr) {
  std::string funcName = expr.funcName;
  if (funcName != "print" || !this->inStateBody || !this->currentFire)
    return std::any();

  mlir::Location loc = builder.getUnknownLoc();

  std::string fmtStr;
  llvm::SmallVector<mlir::Value, 4> subs;
  for (auto &argExpr : expr.args) {
    if (argExpr->getKind() == ast::ASTNodeKind::StringLiteral) {
      std::string raw = static_cast<ast::StringLiteral *>(argExpr.get())->value;
      fmtStr += raw;
    } else {
      auto any = visit(argExpr);
      if (any.has_value() && any.type() == typeid(mlir::Value)) {
        mlir::Value v = std::any_cast<mlir::Value>(any);
        if (v) {
          fmtStr += "%0d";
          subs.push_back(v);
        }
      }
    }
  }
  fmtStr += "\n";

  mlir::Value i1Clk =
      circt::seq::FromClockOp::create(builder, loc, this->currentClock);
  mlir::Value fire = this->currentFire;
  mlir::StringAttr fmtAttr = builder.getStringAttr(fmtStr);
  mlir::Value fd = circt::hw::ConstantOp::create(
      builder, loc, builder.getIntegerType(32), 0x80000001LL);

  circt::sv::AlwaysFFOp::create(
      builder, loc, circt::sv::EventControl::AtPosEdge, i1Clk, [&]() {
        circt::sv::IfOp::create(builder, loc, fire, [&]() {
          circt::sv::FWriteOp::create(builder, loc, fd, fmtAttr, subs);
        });
      });

  return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// Expression / identifier resolution
// ─────────────────────────────────────────────────────────────────────────────

std::optional<mlir::Type> LowerVisitor::getMlirType(const ast::Type &t) {
  if (t.kind == ast::TypeKind::PRIMITIVE) {
    if (t.name == "int")
      return builder.getI32Type();
    if (t.name == "byte")
      return builder.getI8Type();
    if (t.name == "int8" || t.name == "uint8")
      return builder.getI8Type();
    if (t.name == "int16")
      return builder.getIntegerType(16);
    if (t.name == "int32")
      return builder.getI32Type();
    if (t.name == "int64" || t.name == "uint64")
      return builder.getIntegerType(64);
    if (t.name == "uint32")
      return builder.getIntegerType(32);
    if (t.name == "uint16")
      return builder.getIntegerType(16);
    if (t.name == "bool")
      return builder.getI1Type();
    if (t.name == "string")
      return circt::hw::StringType::get(&this->ctx);
  }
  return std::nullopt;
}

std::any LowerVisitor::visitIdent(ast::IdentExpr &expr) {
  std::string name = expr.name;
  if (this->localBindings.count(name))
    return mlir::Value(this->localBindings[name]);
  if (this->varRegs.count(this->currentModuleName) &&
      this->varRegs[this->currentModuleName].count(name))
    return mlir::Value(readVar(name));
  if (this->portValues.count(this->currentModuleName) &&
      this->portValues[this->currentModuleName].count(name))
    return this->portValues[this->currentModuleName][name];
  if (this->enumValues.count(name)) {
    auto [val, w] = this->enumValues[name];
    return mlir::Value(circt::hw::ConstantOp::create(
        builder, builder.getUnknownLoc(), builder.getIntegerType(w), val));
  }
  return std::any();
}

std::any LowerVisitor::visitNumber_literal(ast::NumberLiteral &expr) {
  int64_t value = expr.value;
  mlir::Type t =
      this->currentExprType ? this->currentExprType : builder.getI32Type();
  if (!mlir::isa<mlir::IntegerType>(t))
    t = builder.getI32Type();
  return mlir::Value(circt::hw::ConstantOp::create(
      builder, builder.getUnknownLoc(), t, value));
}

std::any LowerVisitor::visitExpr(ast::Expr &expr) {
  mlir::Location loc = builder.getUnknownLoc();

  if (expr.getKind() == ast::ASTNodeKind::IdentExpr)
    return visitIdent(static_cast<ast::IdentExpr &>(expr));
  if (expr.getKind() == ast::ASTNodeKind::FieldExpr)
    return visitIdent_field(static_cast<ast::FieldExpr &>(expr));
  if (expr.getKind() == ast::ASTNodeKind::IndexExpr)
    return visitIndexExpr(static_cast<ast::IndexExpr &>(expr));
  if (expr.getKind() == ast::ASTNodeKind::NumberLiteral)
    return visitNumber_literal(static_cast<ast::NumberLiteral &>(expr));
  if (expr.getKind() == ast::ASTNodeKind::StringLiteral)
    return std::any();
  if (expr.getKind() == ast::ASTNodeKind::NilLiteral)
    return std::any();
  if (expr.getKind() == ast::ASTNodeKind::CallExpr)
    return visitExpr_func_call(static_cast<ast::CallExpr &>(expr));

  if (expr.getKind() == ast::ASTNodeKind::UnaryExpr) {
    auto &un = static_cast<ast::UnaryExpr &>(expr);
    auto inner = visit(un.expr);
    if (!inner.has_value())
      return std::any();
    mlir::Value v = std::any_cast<mlir::Value>(inner);
    if (un.op == "!") {
      mlir::Value one =
          circt::hw::ConstantOp::create(builder, loc, v.getType(), 1);
      return mlir::Value(circt::comb::XorOp::create(builder, loc, v, one));
    }
    if (un.op == "-") {
      mlir::Value zero =
          circt::hw::ConstantOp::create(builder, loc, v.getType(), 0);
      return mlir::Value(circt::comb::SubOp::create(builder, loc, zero, v));
    }
    notImplemented("unary operator '" + un.op + "'");
  }

  if (expr.getKind() == ast::ASTNodeKind::BinaryExpr) {
    auto &bin = static_cast<ast::BinaryExpr &>(expr);
    auto la = visit(bin.lhs);
    auto ra = visit(bin.rhs);
    if (!la.has_value() || !ra.has_value())
      return std::any();
    mlir::Value lhs = std::any_cast<mlir::Value>(la);
    mlir::Value rhs = std::any_cast<mlir::Value>(ra);
    if (!lhs || !rhs)
      return std::any();

    auto li = mlir::dyn_cast<mlir::IntegerType>(lhs.getType());
    auto ri = mlir::dyn_cast<mlir::IntegerType>(rhs.getType());
    if (li && ri && li.getWidth() != ri.getWidth()) {
      unsigned w = std::max(li.getWidth(), ri.getWidth());
      mlir::Type wide = builder.getIntegerType(w);
      lhs = coerce(lhs, wide, loc);
      rhs = coerce(rhs, wide, loc);
    }

    std::string op = bin.op;
    if (op == "+")
      return mlir::Value(circt::comb::AddOp::create(builder, loc, lhs, rhs));
    if (op == "-")
      return mlir::Value(circt::comb::SubOp::create(builder, loc, lhs, rhs));
    if (op == "*")
      return mlir::Value(circt::comb::MulOp::create(builder, loc, lhs, rhs));
    if (op == "/")
      return mlir::Value(circt::comb::DivUOp::create(builder, loc, lhs, rhs));
    if (op == "%")
      return mlir::Value(circt::comb::ModUOp::create(builder, loc, lhs, rhs));
    if (op == "&")
      return mlir::Value(circt::comb::AndOp::create(builder, loc, lhs, rhs));
    if (op == "|")
      return mlir::Value(circt::comb::OrOp::create(builder, loc, lhs, rhs));
    if (op == "^")
      return mlir::Value(circt::comb::XorOp::create(builder, loc, lhs, rhs));
    if (op == "<<")
      return mlir::Value(circt::comb::ShlOp::create(builder, loc, lhs, rhs));
    if (op == ">>")
      return mlir::Value(circt::comb::ShrUOp::create(builder, loc, lhs, rhs));
    if (op == "&&")
      return mlir::Value(circt::comb::AndOp::create(builder, loc, lhs, rhs));
    if (op == "||")
      return mlir::Value(circt::comb::OrOp::create(builder, loc, lhs, rhs));
    if (op == "==")
      return mlir::Value(circt::comb::ICmpOp::create(
          builder, loc, circt::comb::ICmpPredicate::eq, lhs, rhs));
    if (op == "!=")
      return mlir::Value(circt::comb::ICmpOp::create(
          builder, loc, circt::comb::ICmpPredicate::ne, lhs, rhs));
    if (op == "<")
      return mlir::Value(circt::comb::ICmpOp::create(
          builder, loc, circt::comb::ICmpPredicate::ult, lhs, rhs));
    if (op == "<=")
      return mlir::Value(circt::comb::ICmpOp::create(
          builder, loc, circt::comb::ICmpPredicate::ule, lhs, rhs));
    if (op == ">")
      return mlir::Value(circt::comb::ICmpOp::create(
          builder, loc, circt::comb::ICmpPredicate::ugt, lhs, rhs));
    if (op == ">=")
      return mlir::Value(circt::comb::ICmpOp::create(
          builder, loc, circt::comb::ICmpPredicate::uge, lhs, rhs));
    notImplemented("binary operator '" + op + "'");
  }

  if (expr.getKind() == ast::ASTNodeKind::UpdateExpr) {
    auto &upd = static_cast<ast::UpdateExpr &>(expr);
    auto inner = visit(upd.expr);
    if (!inner.has_value())
      return std::any();
    mlir::Value v = std::any_cast<mlir::Value>(inner);
    if (!v)
      return std::any();

    if (upd.expr->getKind() == ast::ASTNodeKind::IdentExpr) {
      std::string nm = static_cast<ast::IdentExpr *>(upd.expr.get())->name;
      if (this->varRegs[this->currentModuleName].count(nm) &&
          this->currentFire) {
        mlir::Value one =
            circt::hw::ConstantOp::create(builder, loc, v.getType(), 1);
        std::string opTxt = upd.op;
        mlir::Value updated =
            (opTxt == "--")
                ? mlir::Value(circt::comb::SubOp::create(builder, loc, v, one))
                : mlir::Value(circt::comb::AddOp::create(builder, loc, v, one));
        writeVar(nm, updated, this->currentFire);
      }
    } else if (upd.expr->getKind() == ast::ASTNodeKind::IndexExpr &&
               this->currentFire) {
      mlir::Value one =
          circt::hw::ConstantOp::create(builder, loc, v.getType(), 1);
      mlir::Value updated =
          (upd.op == "--")
              ? mlir::Value(circt::comb::SubOp::create(builder, loc, v, one))
              : mlir::Value(circt::comb::AddOp::create(builder, loc, v, one));
      writeArray(upd.expr.get(), "=", updated, this->currentFire);
    }
    return v;
  }

  return std::any();
}

std::any LowerVisitor::visitDecl_var(ast::VarDecl &decl) {
  if (decl.typedIdent.type.kind == ast::TypeKind::ARRAY) {
    std::vector<int64_t> dims;
    const ast::Type *t = &decl.typedIdent.type;
    while (t->kind == ast::TypeKind::ARRAY && t->elementType) {
      dims.push_back(t->size);
      t = t->elementType.get();
    }
    auto elemType = getMlirType(*t);
    if (!elemType) {
      std::cerr << "Error: array element type '" << t->name
                << "' is not supported.\n";
      exit(1);
    }
    if (decl.initExpr) {
      std::cerr << "Error: array initializers are not supported; assign "
                   "elements individually instead.\n";
      exit(1);
    }
    for (auto &id : decl.typedIdent.idents)
      if (!this->arrayInfo[this->currentModuleName].count(id))
        declareArray(id, dims, elemType.value());
    return std::any();
  }

  auto maybeType = getMlirType(decl.typedIdent.type);
  if (!maybeType)
    return std::any();
  mlir::Location loc = builder.getUnknownLoc();
  for (auto &id : decl.typedIdent.idents) {
    std::string name = id;
    if (this->varRegs[this->currentModuleName].count(name))
      continue;
    this->variables[this->currentModuleName][name] = maybeType.value();
    mlir::Value zero =
        circt::hw::ConstantOp::create(builder, loc, maybeType.value(), 0);
    circt::seq::CompRegOp reg = circt::seq::CompRegOp::create(
        builder, loc, zero, this->currentClock, this->currentReset, zero);
    this->varRegs[this->currentModuleName][name] = reg;
    this->varNext[this->currentModuleName][name] = reg.getResult();
  }
  if (decl.initExpr) {
    auto rhs = visit(decl.initExpr);
    if (rhs.has_value() && this->currentFire) {
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

std::any LowerVisitor::visitStmt(ast::Stmt &stmt) { return visit(&stmt); }

std::any LowerVisitor::visitStmt_if(ast::IfStmt &stmt) {
  mlir::Location loc = builder.getUnknownLoc();
  auto condA = visit(stmt.cond);
  if (!condA.has_value())
    return std::any();
  mlir::Value cond = std::any_cast<mlir::Value>(condA);
  if (!cond)
    return std::any();

  if (auto it = mlir::dyn_cast<mlir::IntegerType>(cond.getType()))
    if (it.getWidth() != 1)
      cond = circt::comb::ICmpOp::create(
          builder, loc, circt::comb::ICmpPredicate::ne, cond,
          circt::hw::ConstantOp::create(builder, loc, cond.getType(), 0));

  mlir::Value outerFire = this->currentFire;
  mlir::Value thenFire = outerFire ? mlir::Value(circt::comb::AndOp::create(
                                         builder, loc, outerFire, cond))
                                   : cond;

  if (stmt.thenBranch) {
    this->currentFire = thenFire;
    for (auto &s : stmt.thenBranch->statements)
      visit(s);
    this->currentFire = outerFire;
  }

  mlir::Value notCond = circt::comb::XorOp::create(
      builder, loc, cond,
      circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
  mlir::Value elseFire = outerFire ? mlir::Value(circt::comb::AndOp::create(
                                         builder, loc, outerFire, notCond))
                                   : notCond;

  if (stmt.elseBranch) {
    this->currentFire = elseFire;
    visit(stmt.elseBranch);
    this->currentFire = outerFire;
  }
  return std::any();
}

std::any LowerVisitor::visitStmt_binary(ast::BinaryStmt &stmt) {
  mlir::Location loc = builder.getUnknownLoc();
  std::string op = stmt.op;

  auto lhsExpr = stmt.lhs;
  auto rhsExpr = stmt.rhs;

  std::string lhsName;
  bool lhsIsVar = false;
  bool lhsIsArrayElem = false;
  if (lhsExpr->getKind() == ast::ASTNodeKind::IdentExpr) {
    lhsName = static_cast<ast::IdentExpr *>(lhsExpr.get())->name;
    lhsIsVar = true;
  } else if (lhsExpr->getKind() == ast::ASTNodeKind::IndexExpr) {
    lhsIsArrayElem = true;
  }

  mlir::Type hint;
  if (lhsIsVar && this->variables[this->currentModuleName].count(lhsName))
    hint = this->variables[this->currentModuleName][lhsName];
  if (lhsIsArrayElem) {
    std::string arrName;
    std::vector<ast::Expr *> arrIndices;
    if (peelIndexChain(lhsExpr.get(), arrName, arrIndices) &&
        this->arrayInfo[this->currentModuleName].count(arrName))
      hint = this->arrayInfo[this->currentModuleName][arrName].elemType;
  }
  mlir::Type prevHint = this->currentExprType;
  this->currentExprType = hint;
  auto rhsAny = visit(rhsExpr);
  this->currentExprType = prevHint;
  mlir::Value rhsVal;
  if (rhsAny.has_value() && rhsAny.type() == typeid(mlir::Value))
    rhsVal = std::any_cast<mlir::Value>(rhsAny);

  mlir::Value fire =
      this->currentFire
          ? this->currentFire
          : circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);

  if (op == "<-" || op == "->") {
    bool rev = (op == "->");
    auto srcExpr = rev ? lhsExpr : rhsExpr;
    auto dstExpr = rev ? rhsExpr : lhsExpr;

    std::string dstName;
    std::string dstInst, dstPort;
    bool dstIsIdent = false, dstIsField = false, dstIsArrayElem = false;
    if (dstExpr->getKind() == ast::ASTNodeKind::IdentExpr) {
      dstName = static_cast<ast::IdentExpr *>(dstExpr.get())->name;
      dstIsIdent = true;
    } else if (dstExpr->getKind() == ast::ASTNodeKind::FieldExpr) {
      dstInst = static_cast<ast::FieldExpr *>(dstExpr.get())->object;
      dstPort = static_cast<ast::FieldExpr *>(dstExpr.get())->field;
      dstIsField = true;
    } else if (dstExpr->getKind() == ast::ASTNodeKind::IndexExpr) {
      dstIsArrayElem = true;
    }

    auto srcAny = visit(srcExpr);
    mlir::Value srcVal;
    if (srcAny.has_value() && srcAny.type() == typeid(mlir::Value))
      srcVal = std::any_cast<mlir::Value>(srcAny);

    // ── instantiate block: feed or wire instance input channels ────────────
    if (this->insideInsantiate) {
      if (!dstIsField)
        return std::any();
      if (!this->instances.count(dstInst)) {
        std::cerr << "Error: unknown instance '" << dstInst
                  << "' in instantiate block (declare instances before "
                     "feeding or connecting them).\n";
        exit(1);
      }
      auto &feeds = this->instanceFeeds[dstInst];
      auto portIt = feeds.find(dstPort);
      if (portIt == feeds.end()) {
        std::cerr << "Error: instance '" << dstInst
                  << "' has no input channel named '" << dstPort << "'.\n";
        exit(1);
      }
      if (!srcVal) {
        std::cerr << "Error: cannot evaluate the value fed to '" << dstInst
                  << "." << dstPort
                  << "' (connect instance outputs as <inst>.<port>, with "
                     "both instances declared beforehand).\n";
        exit(1);
      }
      InstanceFeed &feed = portIt->second;
      circt::hw::InstanceOp inst = this->instances[dstInst];

      if (auto chSrc =
              mlir::dyn_cast<mlir::TypedValue<circt::esi::ChannelType>>(
                  srcVal)) {
        // Machine-to-machine connection: b.in <- a.out
        if (feed.fedChannel || feed.fedConstant) {
          std::cerr << "Error: input '" << dstInst << "." << dstPort
                    << "' is already driven; each input channel may have "
                       "only one source.\n";
          exit(1);
        }
        // ESI channels are point-to-point: one consumer per output.
        if (srcExpr->getKind() == ast::ASTNodeKind::FieldExpr) {
          auto *sf = static_cast<ast::FieldExpr *>(srcExpr.get());
          std::pair<std::string, std::string> key = {sf->object, sf->field};
          if (this->connectedOutputs.count(key)) {
            std::cerr << "Error: output '" << sf->object << "." << sf->field
                      << "' is already connected; an output channel may "
                         "feed only one input.\n";
            exit(1);
          }
          this->connectedOutputs.insert(key);
        }
        mlir::Type portTy = inst->getOperand(feed.operandIdx).getType();
        if (chSrc.getType() != portTy) {
          auto srcCh = chSrc.getType();
          auto dstCh = mlir::cast<circt::esi::ChannelType>(portTy);
          std::cerr << "Error: channel type mismatch connecting to '"
                    << dstInst << "." << dstPort << "'";
          auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcCh.getInner());
          auto dstInt = mlir::dyn_cast<mlir::IntegerType>(dstCh.getInner());
          if (srcInt && dstInt)
            std::cerr << ": source is " << srcInt.getWidth()
                      << " bits, destination is " << dstInt.getWidth()
                      << " bits";
          std::cerr << ". Declare both ports with the same type.\n";
          exit(1);
        }
        // Insert a FIFO stage so producer and consumer are decoupled, then
        // swap it in as the instance's input. HW module bodies are graph
        // regions, so the connection is legal regardless of declaration
        // order of the two instances.
        circt::esi::FIFOOp stage = circt::esi::FIFOOp::create(
            builder, loc, portTy, this->currentClock, this->currentReset,
            chSrc, 5);
        inst->setOperand(feed.operandIdx, stage.getResult());
        // The placeholder constant-feed plumbing is now dead; remove it in
        // use-order (fifo uses wrap, wrap uses the regs).
        feed.fifo->erase();
        feed.wrap->erase();
        feed.dataReg->erase();
        feed.validReg->erase();
        feed.fedChannel = true;
        return std::any();
      }

      // Constant/value preload (previous behaviour): patch the feed regs.
      if (feed.fedChannel) {
        std::cerr << "Error: input '" << dstInst << "." << dstPort
                  << "' is already connected to a machine output; it cannot "
                     "also be fed a value.\n";
        exit(1);
      }
      mlir::Type innerTy = feed.dataReg.getResult().getType();
      mlir::Value data = coerce(srcVal, innerTy, loc);
      mlir::Value trueV =
          circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      feed.dataReg->setOperand(0, data);
      feed.dataReg->setOperand(3, data);
      feed.validReg->setOperand(0, trueV);
      feed.validReg->setOperand(3, trueV);
      feed.fedConstant = true;
      return std::any();
    }

    if (srcVal) {
      if (auto chSrc =
              mlir::dyn_cast<mlir::TypedValue<circt::esi::ChannelType>>(
                  srcVal)) {
        mlir::Type inner = chSrc.getType().getInner();
        circt::esi::UnwrapValidReadyOp unwrap =
            circt::esi::UnwrapValidReadyOp::create(builder, loc, chSrc, fire);
        mlir::Value data = unwrap.getRawOutput();
        mlir::Value valid = unwrap.getValid();
        mlir::Value cond =
            circt::comb::AndOp::create(builder, loc, fire, valid);

        if (dstIsIdent) {
          if (!this->varRegs[this->currentModuleName].count(dstName)) {
            mlir::Value zero =
                circt::hw::ConstantOp::create(builder, loc, inner, 0);
            circt::seq::CompRegOp reg = circt::seq::CompRegOp::create(
                builder, loc, zero, this->currentClock, this->currentReset,
                zero);
            this->varRegs[this->currentModuleName][dstName] = reg;
            this->varNext[this->currentModuleName][dstName] = reg.getResult();
            this->variables[this->currentModuleName][dstName] = inner;
          }
          writeVar(dstName, data, cond);
          if (this->inStateHeader) {
            this->currentFire = cond;
            this->localBindings[dstName] = data;
          }
        } else if (dstIsArrayElem) {
          writeArray(dstExpr.get(), "=", data, cond);
          if (this->inStateHeader)
            this->currentFire = cond;
        }
        return std::any();
      }
    }

    if (dstIsIdent && this->portDirs[this->currentModuleName].count(dstName) &&
        this->portDirs[this->currentModuleName][dstName] ==
            circt::hw::ModulePort::Direction::Output &&
        srcVal) {
      mlir::Type portType;
      for (auto &p : this->currentPorts)
        if (p.name == dstName) {
          portType = p.type;
          break;
        }
      auto chTy = mlir::dyn_cast<circt::esi::ChannelType>(portType);
      if (!chTy)
        return std::any();
      mlir::Value zero =
          circt::hw::ConstantOp::create(builder, loc, chTy.getInner(), 0);
      mlir::Value falseV =
          circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
      mlir::Value prevData =
          this->outputDataNext[this->currentModuleName][dstName];
      mlir::Value prevValid =
          this->outputValidNext[this->currentModuleName][dstName];
      if (!prevData)
        prevData = zero;
      if (!prevValid)
        prevValid = falseV;
      mlir::Value coercedSrc = coerce(srcVal, chTy.getInner(), loc);
      mlir::Value newData =
          circt::comb::MuxOp::create(builder, loc, fire, coercedSrc, prevData);
      mlir::Value newValid =
          circt::comb::OrOp::create(builder, loc, fire, prevValid);
      this->outputDataNext[this->currentModuleName][dstName] = newData;
      this->outputValidNext[this->currentModuleName][dstName] = newValid;
      return std::any();
    }

    return std::any();
  }

  if (lhsIsArrayElem && rhsVal) {
    writeArray(lhsExpr.get(), op, rhsVal, fire);
    return std::any();
  }

  if (lhsIsVar && this->varRegs[this->currentModuleName].count(lhsName) &&
      rhsVal) {
    mlir::Value current = readVar(lhsName);
    mlir::Value newVal = applyCompound(op, current, rhsVal, loc);
    writeVar(lhsName, newVal, fire);
    return std::any();
  }

  return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// Instantiation (top-level)
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitDecl_instantiate(ast::InstantiateDecl &decl) {
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
    if (mlir::isa<circt::hw::OutputOp>(op))
      op.erase();
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

std::any LowerVisitor::visitInst_module(ast::InstModuleStmt &stmt) {
  std::string moduleName = stmt.callExpr->funcName;
  std::string varName = stmt.varName;

  if (!this->modules.contains(moduleName)) {
    std::cerr << "Error: module '" << moduleName
              << "' not found for instantiation!\n";
    exit(1);
  }
  circt::hw::HWModuleOp mod = this->modules[moduleName];
  llvm::SmallVector<circt::hw::PortInfo, 4> portInfo = mod.getPortList();
  std::vector<mlir::Value> instanceOperands;

  for (auto port : portInfo) {
    if (port.dir == circt::hw::ModulePort::Direction::Output)
      continue;

    mlir::Type portType = port.type;
    if (auto channelType = mlir::dyn_cast<circt::esi::ChannelType>(portType)) {
      mlir::Type innerType = channelType.getInner();
      mlir::Value dummyData = circt::hw::ConstantOp::create(
          builder, builder.getUnknownLoc(), innerType, 0);
      mlir::Value dummyValid = circt::hw::ConstantOp::create(
          builder, builder.getUnknownLoc(), builder.getI1Type(), 0);
      circt::seq::CompRegOp channel_data = circt::seq::CompRegOp::create(
          builder, builder.getUnknownLoc(), dummyData, this->currentClock,
          this->currentReset, dummyData);
      circt::seq::CompRegOp channel_valid = circt::seq::CompRegOp::create(
          builder, builder.getUnknownLoc(), dummyValid, this->currentClock,
          this->currentReset, dummyValid);
      circt::esi::WrapValidReadyOp vr = circt::esi::WrapValidReadyOp::create(
          builder, builder.getUnknownLoc(), portType, builder.getI1Type(),
          channel_data, channel_valid);
      mlir::TypedValue<circt::esi::ChannelType> ch_in = vr.getChanOutput();
      circt::esi::FIFOOp fifo = circt::esi::FIFOOp::create(
          builder, builder.getUnknownLoc(), port.type, this->currentClock,
          this->currentReset, ch_in, 5);
      InstanceFeed feed;
      feed.dataReg = channel_data;
      feed.validReg = channel_valid;
      feed.wrap = vr;
      feed.fifo = fifo;
      feed.operandIdx = (unsigned)instanceOperands.size();
      this->instanceFeeds[varName][port.name.str()] = feed;
      instanceOperands.push_back(fifo.getResult());
    } else if (mlir::isa<circt::seq::ClockType>(portType)) {
      instanceOperands.push_back(this->currentClock);
    } else if (mlir::isa<mlir::IntegerType>(portType) &&
               mlir::cast<mlir::IntegerType>(portType).getWidth() == 1) {
      instanceOperands.push_back(this->currentReset);
    } else {
      std::cerr << "Error: unsupported port type for port '"
                << port.name.getValue().str() << "' in module '" << moduleName
                << "'!\n";
      exit(1);
    }
  }
  circt::hw::InstanceOp instance = circt::hw::InstanceOp::create(
      builder, builder.getUnknownLoc(), mod, varName, instanceOperands);
  if (this->instances.contains(varName)) {
    std::cerr << "Error: instance '" << varName << "' is already declared!\n";
    exit(1);
  }
  this->instances[varName] = instance;
  return std::any();
}

std::any LowerVisitor::visitIdent_field(ast::FieldExpr &expr) {
  if (!this->insideInsantiate) {
    return std::any();
  }
  auto inst = expr.object;
  auto port = expr.field;
  if (this->instances.count(inst)) {
    circt::hw::InstanceOp op = this->instances[inst];
    auto plist = op.getPortList();
    unsigned outIdx = 0;
    for (auto &p : plist) {
      if (p.dir == circt::hw::ModulePort::Direction::Output && p.name == port)
        return mlir::Value(op.getResult(outIdx));
      if (p.dir == circt::hw::ModulePort::Direction::Output)
        ++outIdx;
    }
  }
  return std::any();
}

// ─────────────────────────────────────────────────────────────────────────────
// Dummy / Unimplemented declarations and statements
// ─────────────────────────────────────────────────────────────────────────────

std::any LowerVisitor::visitBlockStmt(ast::BlockStmt &stmt) {
  for (auto &s : stmt.statements)
    visit(s);
  return std::any();
}

std::optional<int64_t> LowerVisitor::evalConst(ast::Expr *expr, const std::map<std::string, int64_t> &consts) {
  if (!expr) return std::nullopt;
  if (expr->getKind() == ast::ASTNodeKind::NumberLiteral) {
    return static_cast<ast::NumberLiteral*>(expr)->value;
  }
  if (expr->getKind() == ast::ASTNodeKind::IdentExpr) {
    std::string name = static_cast<ast::IdentExpr*>(expr)->name;
    if (consts.count(name)) return consts.at(name);
    if (this->enumValues.count(name)) return this->enumValues.at(name).first;
    return std::nullopt;
  }
  if (expr->getKind() == ast::ASTNodeKind::BinaryExpr) {
    auto bin = static_cast<ast::BinaryExpr*>(expr);
    auto l = evalConst(bin->lhs.get(), consts);
    auto r = evalConst(bin->rhs.get(), consts);
    if (l && r) {
      std::string op = bin->op;
      if (op == "+") return *l + *r;
      if (op == "-") return *l - *r;
      if (op == "*") return *l * *r;
      if (op == "/") return *r != 0 ? std::optional<int64_t>(*l / *r) : std::nullopt;
      if (op == "%") return *r != 0 ? std::optional<int64_t>(*l % *r) : std::nullopt;
      if (op == "<<") return *l << *r;
      if (op == ">>") return *l >> *r;
      if (op == "&") return *l & *r;
      if (op == "|") return *l | *r;
      if (op == "^") return *l ^ *r;
      if (op == "<") return *l < *r;
      if (op == "<=") return *l <= *r;
      if (op == ">") return *l > *r;
      if (op == ">=") return *l >= *r;
      if (op == "==") return *l == *r;
      if (op == "!=") return *l != *r;
      if (op == "&&") return *l && *r;
      if (op == "||") return *l || *r;
    }
  }
  if (expr->getKind() == ast::ASTNodeKind::UnaryExpr) {
    auto un = static_cast<ast::UnaryExpr*>(expr);
    auto inner = evalConst(un->expr.get(), consts);
    if (inner) {
      if (un->op == "!") return !(*inner);
      if (un->op == "-") return -(*inner);
    }
  }
  return std::nullopt;
}

std::any LowerVisitor::visitForStmt(ast::ForStmt &stmt) {
  mlir::Location loc = builder.getUnknownLoc();

  std::string loopVar;
  int64_t initVal = 0;
  bool hasInit = false;
  mlir::Type loopVarType = builder.getI32Type();

  // The init expression may reference enclosing loop variables
  // (e.g. `for (var int j = i; ...)` inside an outer loop over i).
  if (stmt.init) {
    if (stmt.init->getKind() == ast::ASTNodeKind::VarDeclStmt) {
      auto vd = static_cast<ast::VarDeclStmt*>(stmt.init.get())->varDecl;
      if (vd && !vd->typedIdent.idents.empty()) {
        loopVar = vd->typedIdent.idents[0];
        if (auto t = getMlirType(vd->typedIdent.type))
          if (mlir::isa<mlir::IntegerType>(*t))
            loopVarType = *t;
        if (vd->initExpr) {
          auto v = evalConst(vd->initExpr.get(), this->loopConstBindings);
          if (v) {
            initVal = *v;
            hasInit = true;
          }
        }
      }
    } else if (stmt.init->getKind() == ast::ASTNodeKind::BinaryStmt) {
      auto bs = static_cast<ast::BinaryStmt*>(stmt.init.get());
      if (bs->op == "=" && bs->lhs->getKind() == ast::ASTNodeKind::IdentExpr) {
        loopVar = static_cast<ast::IdentExpr*>(bs->lhs.get())->name;
        if (this->variables[this->currentModuleName].count(loopVar)) {
          mlir::Type t = this->variables[this->currentModuleName][loopVar];
          if (mlir::isa<mlir::IntegerType>(t))
            loopVarType = t;
        }
        auto v = evalConst(bs->rhs.get(), this->loopConstBindings);
        if (v) {
          initVal = *v;
          hasInit = true;
        }
      }
    }
  }

  if (!hasInit || loopVar.empty()) {
    std::cerr << "Error: for loop initialization must declare or assign a compile-time constant variable.\n";
    exit(1);
  }

  // Save any outer binding of the same name (loop-variable shadowing).
  mlir::Value oldVal;
  bool hadOldVal = this->localBindings.count(loopVar);
  if (hadOldVal) oldVal = this->localBindings[loopVar];
  bool hadOldConst = this->loopConstBindings.count(loopVar);
  int64_t oldConst = hadOldConst ? this->loopConstBindings[loopVar] : 0;

  int64_t currVal = initVal;
  const int maxIterations = 100000; // Safeguard against infinite unrolling loops
  int iterations = 0;

  this->forLoopDepth++;

  while (true) {
    if (iterations >= maxIterations) {
      std::cerr << "Error: for loop reached iteration limit of " << maxIterations << " (infinite loop detected).\n";
      exit(1);
    }

    this->loopConstBindings[loopVar] = currVal;

    // Evaluate condition
    if (stmt.cond) {
      auto condVal = evalConst(stmt.cond.get(), this->loopConstBindings);
      if (!condVal) {
        std::cerr << "Error: for loop bound must be a compile-time constant "
                     "(a literal or an enclosing loop variable). For a "
                     "runtime bound, use a state that increments a shared "
                     "variable and loops with goto.\n";
        exit(1);
      }
      if (*condVal == 0) {
        break; // Condition is false, terminate loop unrolling
      }
    } else {
      // Unconditional loop is infinite unrolling in hardware, which is a compile error
      std::cerr << "Error: for loop must have a compile-time conditional bound.\n";
      exit(1);
    }

    // Set loop variable inside MLIR lowering environment as a constant
    mlir::Value currValVal = circt::hw::ConstantOp::create(
        builder, loc, loopVarType, currVal);
    this->localBindings[loopVar] = currValVal;

    // Lower the body
    if (stmt.body) {
      visitBlockStmt(*stmt.body);
    }

    // Update loop variable
    bool hasUpdate = false;
    if (stmt.update) {
      if (stmt.update->getKind() == ast::ASTNodeKind::ExprStmt) {
        auto expr = static_cast<ast::ExprStmt*>(stmt.update.get())->expr;
        if (expr->getKind() == ast::ASTNodeKind::UpdateExpr) {
          auto ue = static_cast<ast::UpdateExpr*>(expr.get());
          if (ue->expr->getKind() == ast::ASTNodeKind::IdentExpr) {
            std::string name = static_cast<ast::IdentExpr*>(ue->expr.get())->name;
            if (name == loopVar) {
              if (ue->op == "++") {
                currVal++;
                hasUpdate = true;
              } else if (ue->op == "--") {
                currVal--;
                hasUpdate = true;
              }
            }
          }
        }
      } else if (stmt.update->getKind() == ast::ASTNodeKind::BinaryStmt) {
        auto bs = static_cast<ast::BinaryStmt*>(stmt.update.get());
        if (bs->lhs->getKind() == ast::ASTNodeKind::IdentExpr) {
          std::string name = static_cast<ast::IdentExpr*>(bs->lhs.get())->name;
          if (name == loopVar) {
            auto rhsVal = evalConst(bs->rhs.get(), this->loopConstBindings);
            if (rhsVal) {
              const std::string &uop = bs->op;
              if (uop == "=" || uop == ":=") {
                currVal = *rhsVal;
                hasUpdate = true;
              } else if (uop == "+=") {
                currVal += *rhsVal;
                hasUpdate = true;
              } else if (uop == "-=") {
                currVal -= *rhsVal;
                hasUpdate = true;
              } else if (uop == "*=") {
                currVal *= *rhsVal;
                hasUpdate = true;
              } else if (uop == "/=" && *rhsVal != 0) {
                currVal /= *rhsVal;
                hasUpdate = true;
              } else if (uop == "<<=") {
                currVal <<= *rhsVal;
                hasUpdate = true;
              } else if (uop == ">>=") {
                currVal >>= *rhsVal;
                hasUpdate = true;
              }
            }
          }
        }
      }
    }

    if (!hasUpdate) {
      std::cerr << "Error: for loop update must be a compile-time constant increment/decrement on the loop variable.\n";
      exit(1);
    }

    iterations++;
  }

  this->forLoopDepth--;

  // Restore the loop variable in the outer scope
  if (hadOldVal) {
    this->localBindings[loopVar] = oldVal;
  } else {
    this->localBindings.erase(loopVar);
  }
  if (hadOldConst) {
    this->loopConstBindings[loopVar] = oldConst;
  } else {
    this->loopConstBindings.erase(loopVar);
  }

  // When the loop variable names a shared register (`for (i = 0; ...)` with
  // shared i), commit the final value so code after the loop sees it.
  if (this->varRegs[this->currentModuleName].count(loopVar) &&
      this->currentFire) {
    mlir::Type regTy =
        this->varRegs[this->currentModuleName][loopVar].getResult().getType();
    mlir::Value finalVal =
        circt::hw::ConstantOp::create(builder, loc, regTy, currVal);
    writeVar(loopVar, finalVal, this->currentFire);
  }

  return std::any();
}

std::any LowerVisitor::visitSwitchStmt(ast::SwitchStmt &stmt) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Value targetVal;
  if (stmt.target) {
    if (stmt.target->getKind() == ast::ASTNodeKind::ExprStmt) {
      auto exprA = visit(static_cast<ast::ExprStmt*>(stmt.target.get())->expr);
      if (exprA.has_value() && exprA.type() == typeid(mlir::Value)) {
        targetVal = std::any_cast<mlir::Value>(exprA);
      }
    }
  }

  if (!targetVal) {
    // Fallback if target cannot be evaluated to a value
    for (auto &c : stmt.cases) {
      for (auto &s : c.body) {
        visit(s);
      }
    }
    return std::any();
  }

  mlir::Value outerFire = this->currentFire;
  mlir::Value hasMatched = circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);

  // Visit non-default cases
  for (auto &c : stmt.cases) {
    if (c.isDefault) continue;
    auto caseExprA = visit(c.expr);
    if (!caseExprA.has_value() || caseExprA.type() != typeid(mlir::Value)) continue;
    mlir::Value caseVal = std::any_cast<mlir::Value>(caseExprA);
    if (!caseVal) continue;

    // Coerce if types differ
    mlir::Value lhs = targetVal;
    mlir::Value rhs = caseVal;
    auto li = mlir::dyn_cast<mlir::IntegerType>(lhs.getType());
    auto ri = mlir::dyn_cast<mlir::IntegerType>(rhs.getType());
    if (li && ri && li.getWidth() != ri.getWidth()) {
      unsigned w = std::max(li.getWidth(), ri.getWidth());
      mlir::Type wide = builder.getIntegerType(w);
      lhs = coerce(lhs, wide, loc);
      rhs = coerce(rhs, wide, loc);
    }

    mlir::Value eq = circt::comb::ICmpOp::create(builder, loc, circt::comb::ICmpPredicate::eq, lhs, rhs);
    mlir::Value caseFire = outerFire ? mlir::Value(circt::comb::AndOp::create(builder, loc, outerFire, eq)) : eq;

    this->currentFire = caseFire;
    for (auto &s : c.body) {
      visit(s);
    }
    this->currentFire = outerFire;

    hasMatched = circt::comb::OrOp::create(builder, loc, hasMatched, eq);
  }

  // Visit default case
  for (auto &c : stmt.cases) {
    if (!c.isDefault) continue;
    mlir::Value notMatched = circt::comb::XorOp::create(
        builder, loc, hasMatched,
        circt::hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
    mlir::Value defaultFire = outerFire ? mlir::Value(circt::comb::AndOp::create(builder, loc, outerFire, notMatched)) : notMatched;

    this->currentFire = defaultFire;
    for (auto &s : c.body) {
      visit(s);
    }
    this->currentFire = outerFire;
  }

  return std::any();
}

std::any LowerVisitor::visitReturnStmt(ast::ReturnStmt &) {
  // Functions are not supported by the parser (skipped), so ReturnStmt in states is a no-op.
  return std::any();
}
