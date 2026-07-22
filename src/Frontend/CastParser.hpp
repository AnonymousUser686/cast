#pragma once
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "CastAST.hpp"

enum class TokenType {
    // Keywords
    MACHINE, INTERFACE, SHARED, STATES, GOTO, VAR, INSTANTIATE, ENUM, IF, ELSE, INPUT, OUTPUT,
    
    // Types
    T_STRING, T_INT, T_FLOAT, T_BYTE, T_INT32, T_UINT32, T_UINT16, T_BOOL,
    T_INT8, T_UINT8, T_INT16, T_INT64, T_UINT64,
    
    // Literals
    CNAME, INTEGER, HEX, STRING_LIT, NIL,
    
    // Operators & Punctuation
    ARROW_L, ARROW_R, COLON_EQUAL, EQUAL, PLUS_EQUAL, MINUS_EQUAL, STAR_EQUAL, SLASH_EQUAL, XOR_EQUAL, SHL_EQUAL, SHR_EQUAL,
    PLUS_PLUS, MINUS_MINUS,
    PLUS, MINUS, STAR, SLASH, AMP_AMP, PIPE_PIPE, EXCL_EQUAL, EQUAL_EQUAL, GREATER_EQUAL, LESS_EQUAL, GREATER, LESS, PERCENT,
    SHL, SHR, PIPE, AMP, XOR, EXCL,
    LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET, COMMA, SEMICOLON, COLON, DOT,
    
    // Special
    TOK_EOF, TOK_ERROR
};

struct Token {
    TokenType type;
    std::string text;
    size_t line;
    size_t column;
};

class Lexer {
public:
    Lexer(std::string source);
    Token nextToken();

private:
    char peek();
    char next();

    std::string src;
    size_t pos = 0;
    size_t line = 1;
    size_t col = 1;
};

class CastParser {
public:
    CastParser(std::string source);
    std::shared_ptr<ast::Program> parseProgram();

private:
    // Token access
    Token peekToken(size_t offset = 0);
    void consume();
    bool match(TokenType type);
    bool check(TokenType type);
    void expect(TokenType type, const std::string &errMsg);

    // Parsing methods
    std::shared_ptr<ast::Decl> parseDecl();
    std::shared_ptr<ast::EnumDecl> parseEnumDecl();
    std::shared_ptr<ast::MachineDecl> parseMachineDecl();
    std::shared_ptr<ast::InterfaceBlock> parseInterfaceBlock();
    ast::IODecl parseIODecl();
    std::shared_ptr<ast::SharedBlock> parseSharedBlock();
    std::shared_ptr<ast::VarDecl> parseVarDecl();
    std::shared_ptr<ast::StatesBlock> parseStatesBlock();
    std::shared_ptr<ast::StateDecl> parseStateDecl();
    std::shared_ptr<ast::ExceptionDecl> parseExceptionDecl();
    std::shared_ptr<ast::InstantiateDecl> parseInstantiateDecl();
    
    // Statements
    std::shared_ptr<ast::Stmt> parseStmt();
    std::shared_ptr<ast::BlockStmt> parseBlockStmt();
    std::shared_ptr<ast::IfStmt> parseIfStmt();
    std::shared_ptr<ast::ForStmt> parseForStmt();
    std::shared_ptr<ast::SwitchStmt> parseSwitchStmt();
    std::shared_ptr<ast::ReturnStmt> parseReturnStmt();
    std::shared_ptr<ast::GotoStmt> parseGotoStmt();
    std::shared_ptr<ast::Stmt> parseVarDeclOrExprOrAssign();
    std::shared_ptr<ast::Stmt> parseForUpdate();
    
    // Expressions
    std::shared_ptr<ast::Expr> parseExpr();
    std::shared_ptr<ast::Expr> parseBinaryExpr(int precedence = 0);
    std::shared_ptr<ast::Expr> parseUnaryExpr();
    std::shared_ptr<ast::Expr> parsePrimaryExpr();
    std::shared_ptr<ast::Expr> parsePostfixExpr(std::shared_ptr<ast::Expr> base);
    
    // Types and IdentTyped
    ast::Type parseType();
    ast::IdentTyped parseIdentTyped();
    
    Lexer lexer;
    std::vector<Token> buffer;
    size_t bufferPos = 0;
};
