#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <utility>

namespace ast {

enum class ASTNodeKind {
    Program,
    EnumDecl,
    MachineDecl,
    InterfaceBlock,
    SharedBlock,
    StatesBlock,
    StateDecl,
    ExceptionDecl,
    InstantiateDecl,
    VarDecl,
    
    // Statements
    BlockStmt,
    VarDeclStmt,
    InstModuleStmt,
    ForStmt,
    IfStmt,
    SwitchStmt,
    BinaryStmt,
    ReturnStmt,
    ExprStmt,
    GotoStmt,
    
    // Expressions
    IdentExpr,
    FieldExpr,
    IndexExpr,
    NumberLiteral,
    StringLiteral,
    NilLiteral,
    UnaryExpr,
    BinaryExpr,
    UpdateExpr,
    CallExpr
};

struct ASTNode {
    virtual ~ASTNode() = default;
    virtual ASTNodeKind getKind() const = 0;
};

// Types representation
enum class TypeKind {
    PRIMITIVE,
    CUSTOM,
    ARRAY,
    SLICE,
    MAP,
    CHAN
};

struct Type {
    TypeKind kind = TypeKind::PRIMITIVE;
    std::string name; // "uint16", "bool", or enum/custom name "commands"
    std::shared_ptr<Type> elementType;
    std::shared_ptr<Type> keyType;
    int64_t size = 0; // array size
    std::string chanDir; // "", "<-", "->"
};

struct IdentTyped {
    Type type;
    std::vector<std::string> idents;
};

// Expressions
struct Expr : public ASTNode {
};

struct IdentExpr : public Expr {
    std::string name;
    ASTNodeKind getKind() const override { return ASTNodeKind::IdentExpr; }
};

struct FieldExpr : public Expr {
    std::string object;
    std::string field;
    ASTNodeKind getKind() const override { return ASTNodeKind::FieldExpr; }
};

// Array element access: base[index]. Multi-dimensional access chains
// (a[i][j]) nest, with the outermost node holding the last index.
struct IndexExpr : public Expr {
    std::shared_ptr<Expr> base;
    std::shared_ptr<Expr> index;
    ASTNodeKind getKind() const override { return ASTNodeKind::IndexExpr; }
};

struct NumberLiteral : public Expr {
    int64_t value = 0;
    std::string text;
    ASTNodeKind getKind() const override { return ASTNodeKind::NumberLiteral; }
};

struct StringLiteral : public Expr {
    std::string value;
    ASTNodeKind getKind() const override { return ASTNodeKind::StringLiteral; }
};

struct NilLiteral : public Expr {
    ASTNodeKind getKind() const override { return ASTNodeKind::NilLiteral; }
};

struct UnaryExpr : public Expr {
    std::string op;
    std::shared_ptr<Expr> expr;
    ASTNodeKind getKind() const override { return ASTNodeKind::UnaryExpr; }
};

struct BinaryExpr : public Expr {
    std::shared_ptr<Expr> lhs;
    std::string op;
    std::shared_ptr<Expr> rhs;
    ASTNodeKind getKind() const override { return ASTNodeKind::BinaryExpr; }
};

struct UpdateExpr : public Expr {
    std::shared_ptr<Expr> expr;
    std::string op; // "++", "--"
    ASTNodeKind getKind() const override { return ASTNodeKind::UpdateExpr; }
};

struct CallExpr : public Expr {
    std::string funcName;
    std::vector<std::shared_ptr<Expr>> args;
    // For compile-time params if any (e.g. mem_size=1024)
    std::vector<std::pair<std::string, std::shared_ptr<Expr>>> compileTimeParams;
    ASTNodeKind getKind() const override { return ASTNodeKind::CallExpr; }
};

// Statements
struct Stmt : public ASTNode {
};

struct BlockStmt : public Stmt {
    std::vector<std::shared_ptr<Stmt>> statements;
    ASTNodeKind getKind() const override { return ASTNodeKind::BlockStmt; }
};

struct VarDecl : public ASTNode {
    IdentTyped typedIdent;
    std::shared_ptr<Expr> initExpr;
    ASTNodeKind getKind() const override { return ASTNodeKind::VarDecl; }
};

struct VarDeclStmt : public Stmt {
    std::shared_ptr<VarDecl> varDecl;
    ASTNodeKind getKind() const override { return ASTNodeKind::VarDeclStmt; }
};

struct InstModuleStmt : public Stmt {
    std::string varName;
    std::shared_ptr<CallExpr> callExpr;
    ASTNodeKind getKind() const override { return ASTNodeKind::InstModuleStmt; }
};

struct ForStmt : public Stmt {
    std::shared_ptr<Stmt> init; // VarDeclStmt or BinaryStmt
    std::shared_ptr<Expr> cond;
    std::shared_ptr<Stmt> update; // BinaryStmt or ExprStmt wrapping UpdateExpr
    std::shared_ptr<BlockStmt> body;
    ASTNodeKind getKind() const override { return ASTNodeKind::ForStmt; }
};

struct IfStmt : public Stmt {
    std::shared_ptr<Expr> cond;
    std::shared_ptr<BlockStmt> thenBranch;
    std::shared_ptr<Stmt> elseBranch; // BlockStmt or IfStmt
    ASTNodeKind getKind() const override { return ASTNodeKind::IfStmt; }
};

struct SwitchStmt : public Stmt {
    std::shared_ptr<Stmt> target;
    struct Case {
        bool isDefault = false;
        Type type; // case type
        std::shared_ptr<Expr> expr; // case expr
        std::vector<std::shared_ptr<Stmt>> body;
    };
    std::vector<Case> cases;
    ASTNodeKind getKind() const override { return ASTNodeKind::SwitchStmt; }
};

struct BinaryStmt : public Stmt {
    std::shared_ptr<Expr> lhs;
    std::string op; // e.g. "=", "<-", "->", "+=", etc.
    std::shared_ptr<Expr> rhs;
    ASTNodeKind getKind() const override { return ASTNodeKind::BinaryStmt; }
};

struct ReturnStmt : public Stmt {
    std::vector<std::shared_ptr<Expr>> exprs;
    ASTNodeKind getKind() const override { return ASTNodeKind::ReturnStmt; }
};

struct ExprStmt : public Stmt {
    std::shared_ptr<Expr> expr;
    ASTNodeKind getKind() const override { return ASTNodeKind::ExprStmt; }
};

struct GotoStmt : public Stmt {
    std::string targetState;
    ASTNodeKind getKind() const override { return ASTNodeKind::GotoStmt; }
};

// Declarations
struct Decl : public ASTNode {
};

struct EnumDecl : public Decl {
    std::string name;
    std::vector<std::string> members;
    ASTNodeKind getKind() const override { return ASTNodeKind::EnumDecl; }
};

struct IODecl {
    std::string direction; // "input" or "output"
    IdentTyped typedIdent;
    std::vector<std::string> idents; // extra identifiers
};

struct InterfaceBlock : public ASTNode {
    std::vector<IODecl> ioDecls;
    ASTNodeKind getKind() const override { return ASTNodeKind::InterfaceBlock; }
};

struct SharedBlock : public ASTNode {
    std::vector<std::shared_ptr<VarDecl>> varDecls;
    ASTNodeKind getKind() const override { return ASTNodeKind::SharedBlock; }
};

struct StateDecl : public ASTNode {
    std::string name;
    std::vector<std::shared_ptr<BinaryStmt>> headerReceives;
    std::shared_ptr<BlockStmt> body;
    ASTNodeKind getKind() const override { return ASTNodeKind::StateDecl; }
};

struct ExceptionDecl : public ASTNode {
    std::string name;
    std::vector<std::shared_ptr<BinaryStmt>> headerReceives;
    std::shared_ptr<BlockStmt> body;
    ASTNodeKind getKind() const override { return ASTNodeKind::ExceptionDecl; }
};

struct StatesBlock : public ASTNode {
    std::vector<std::shared_ptr<StateDecl>> stateDecls;
    std::vector<std::shared_ptr<ExceptionDecl>> exceptionDecls;
    ASTNodeKind getKind() const override { return ASTNodeKind::StatesBlock; }
};

struct MachineDecl : public Decl {
    std::string name;
    std::vector<std::pair<std::string, Type>> compileTimeParams;
    std::shared_ptr<InterfaceBlock> interfaceBlock;
    std::shared_ptr<SharedBlock> sharedBlock;
    std::shared_ptr<StatesBlock> statesBlock;
    ASTNodeKind getKind() const override { return ASTNodeKind::MachineDecl; }
};

struct InstantiateDecl : public Decl {
    std::shared_ptr<BlockStmt> body;
    ASTNodeKind getKind() const override { return ASTNodeKind::InstantiateDecl; }
};

struct Program : public ASTNode {
    std::vector<std::shared_ptr<Decl>> decls;
    ASTNodeKind getKind() const override { return ASTNodeKind::Program; }
};

} // namespace ast
