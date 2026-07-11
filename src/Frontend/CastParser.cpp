#include "CastParser.hpp"
#include <cctype>
#include <unordered_map>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// Lexer Implementation
// ─────────────────────────────────────────────────────────────────────────────

Lexer::Lexer(std::string source) : src(std::move(source)) {}

char Lexer::peek() {
    if (pos >= src.size()) return '\0';
    return src[pos];
}

char Lexer::next() {
    if (pos >= src.size()) return '\0';
    char c = src[pos++];
    if (c == '\n') {
        line++;
        col = 1;
    } else {
        col++;
    }
    return c;
}

Token Lexer::nextToken() {
    while (true) {
        char c = peek();
        if (c == '\0') {
            return {TokenType::TOK_EOF, "", line, col};
        }
        
        // Whitespace
        if (std::isspace(c)) {
            next();
            continue;
        }
        
        // Comments
        if (c == '/' && pos + 1 < src.size() && src[pos + 1] == '/') {
            // Line comment
            while (peek() != '\n' && peek() != '\0') {
                next();
            }
            continue;
        }
        if (c == '/' && pos + 1 < src.size() && src[pos + 1] == '*') {
            // Block comment
            next(); next(); // consume /*
            while (peek() != '\0') {
                if (peek() == '*' && pos + 1 < src.size() && src[pos + 1] == '/') {
                    next(); next(); // consume */
                    break;
                }
                next();
            }
            continue;
        }
        
        // Hexadecimal / Decimal Integer
        if (std::isdigit(c)) {
            size_t startLine = line;
            size_t startCol = col;
            std::string text;
            if (c == '0' && pos + 1 < src.size() && (src[pos + 1] == 'x' || src[pos + 1] == 'X')) {
                text += next(); // '0'
                text += next(); // 'x'
                while (std::isxdigit(peek())) {
                    text += next();
                }
                return {TokenType::HEX, text, startLine, startCol};
            }
            while (std::isdigit(peek())) {
                text += next();
            }
            return {TokenType::INTEGER, text, startLine, startCol};
        }
        
        // String Literals
        if (c == '"') {
            size_t startLine = line;
            size_t startCol = col;
            std::string text;
            text += next(); // consume '"'
            while (peek() != '\0' && peek() != '"') {
                if (peek() == '\\') {
                    text += next();
                    if (peek() != '\0') text += next();
                } else {
                    text += next();
                }
            }
            if (peek() == '"') {
                text += next();
            }
            return {TokenType::STRING_LIT, text, startLine, startCol};
        }
        
        // Operators & Punctuation
        size_t startLine = line;
        size_t startCol = col;
        
        if (c == '<') {
            next();
            if (peek() == '-') {
                next();
                return {TokenType::ARROW_L, "<-", startLine, startCol};
            }
            if (peek() == '<') {
                next();
                if (peek() == '=') {
                    next();
                    return {TokenType::SHL_EQUAL, "<<=", startLine, startCol};
                }
                return {TokenType::SHL, "<<", startLine, startCol};
            }
            if (peek() == '=') {
                next();
                return {TokenType::LESS_EQUAL, "<=", startLine, startCol};
            }
            return {TokenType::LESS, "<", startLine, startCol};
        }
        if (c == '>') {
            next();
            if (peek() == '>') {
                next();
                if (peek() == '=') {
                    next();
                    return {TokenType::SHR_EQUAL, ">>=", startLine, startCol};
                }
                return {TokenType::SHR, ">>", startLine, startCol};
            }
            if (peek() == '=') {
                next();
                return {TokenType::GREATER_EQUAL, ">=", startLine, startCol};
            }
            return {TokenType::GREATER, ">", startLine, startCol};
        }
        if (c == ':') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::COLON_EQUAL, ":=", startLine, startCol};
            }
            return {TokenType::COLON, ":", startLine, startCol};
        }
        if (c == '=') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::EQUAL_EQUAL, "==", startLine, startCol};
            }
            return {TokenType::EQUAL, "=", startLine, startCol};
        }
        if (c == '+') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::PLUS_EQUAL, "+=", startLine, startCol};
            }
            if (peek() == '+') {
                next();
                return {TokenType::PLUS_PLUS, "++", startLine, startCol};
            }
            return {TokenType::PLUS, "+", startLine, startCol};
        }
        if (c == '-') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::MINUS_EQUAL, "-=", startLine, startCol};
            }
            if (peek() == '-') {
                next();
                return {TokenType::MINUS_MINUS, "--", startLine, startCol};
            }
            if (peek() == '>') {
                next();
                return {TokenType::ARROW_R, "->", startLine, startCol};
            }
            return {TokenType::MINUS, "-", startLine, startCol};
        }
        if (c == '*') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::STAR_EQUAL, "*=", startLine, startCol};
            }
            return {TokenType::STAR, "*", startLine, startCol};
        }
        if (c == '/') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::SLASH_EQUAL, "/=", startLine, startCol};
            }
            return {TokenType::SLASH, "/", startLine, startCol};
        }
        if (c == '^') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::XOR_EQUAL, "^=", startLine, startCol};
            }
            return {TokenType::XOR, "^", startLine, startCol};
        }
        if (c == '!') {
            next();
            if (peek() == '=') {
                next();
                return {TokenType::EXCL_EQUAL, "!=", startLine, startCol};
            }
            return {TokenType::EXCL, "!", startLine, startCol};
        }
        if (c == '&') {
            next();
            if (peek() == '&') {
                next();
                return {TokenType::AMP_AMP, "&&", startLine, startCol};
            }
            return {TokenType::AMP, "&", startLine, startCol};
        }
        if (c == '|') {
            next();
            if (peek() == '|') {
                next();
                return {TokenType::PIPE_PIPE, "||", startLine, startCol};
            }
            return {TokenType::PIPE, "|", startLine, startCol};
        }
        if (c == '%') { next(); return {TokenType::PERCENT, "%", startLine, startCol}; }
        if (c == '(') { next(); return {TokenType::LPAREN, "(", startLine, startCol}; }
        if (c == ')') { next(); return {TokenType::RPAREN, ")", startLine, startCol}; }
        if (c == '{') { next(); return {TokenType::LBRACE, "{", startLine, startCol}; }
        if (c == '}') { next(); return {TokenType::RBRACE, "}", startLine, startCol}; }
        if (c == '[') { next(); return {TokenType::LBRACKET, "[", startLine, startCol}; }
        if (c == ']') { next(); return {TokenType::RBRACKET, "]", startLine, startCol}; }
        if (c == ',') { next(); return {TokenType::COMMA, ",", startLine, startCol}; }
        if (c == ';') { next(); return {TokenType::SEMICOLON, ";", startLine, startCol}; }
        if (c == '.') { next(); return {TokenType::DOT, ".", startLine, startCol}; }
        
        // Identifiers & Keywords
        if (std::isalpha(c) || c == '_') {
            std::string text;
            while (std::isalnum(peek()) || peek() == '_') {
                text += next();
            }
            
            static const std::unordered_map<std::string, TokenType> keywords = {
                {"machine", TokenType::MACHINE},
                {"interface", TokenType::INTERFACE},
                {"shared", TokenType::SHARED},
                {"states", TokenType::STATES},
                {"goto", TokenType::GOTO},
                {"var", TokenType::VAR},
                {"instantiate", TokenType::INSTANTIATE},
                {"enum", TokenType::ENUM},
                {"if", TokenType::IF},
                {"else", TokenType::ELSE},
                {"input", TokenType::INPUT},
                {"output", TokenType::OUTPUT},
                {"string", TokenType::T_STRING},
                {"int", TokenType::T_INT},
                {"float", TokenType::T_FLOAT},
                {"byte", TokenType::T_BYTE},
                {"int32", TokenType::T_INT32},
                {"uint32", TokenType::T_UINT32},
                {"uint16", TokenType::T_UINT16},
                {"bool", TokenType::T_BOOL},
                {"nil", TokenType::NIL}
            };
            
            auto it = keywords.find(text);
            if (it != keywords.end()) {
                return {it->second, text, startLine, startCol};
            }
            return {TokenType::CNAME, text, startLine, startCol};
        }
        
        std::string errText(1, next());
        return {TokenType::TOK_ERROR, errText, startLine, startCol};
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CastParser Implementation
// ─────────────────────────────────────────────────────────────────────────────

static int getPrecedence(TokenType type) {
    switch (type) {
        case TokenType::PIPE_PIPE: return 1;
        case TokenType::AMP_AMP: return 2;
        case TokenType::PIPE: return 3;
        case TokenType::XOR: return 4;
        case TokenType::AMP: return 5;
        case TokenType::EQUAL_EQUAL:
        case TokenType::EXCL_EQUAL: return 6;
        case TokenType::LESS:
        case TokenType::LESS_EQUAL:
        case TokenType::GREATER:
        case TokenType::GREATER_EQUAL: return 7;
        case TokenType::SHL:
        case TokenType::SHR: return 8;
        case TokenType::PLUS:
        case TokenType::MINUS: return 9;
        case TokenType::STAR:
        case TokenType::SLASH:
        case TokenType::PERCENT: return 10;
        default: return -1;
    }
}

CastParser::CastParser(std::string source) : lexer(std::move(source)) {}

Token CastParser::peekToken(size_t offset) {
    while (buffer.size() <= bufferPos + offset) {
        buffer.push_back(lexer.nextToken());
    }
    return buffer[bufferPos + offset];
}

void CastParser::consume() {
    peekToken(0);
    bufferPos++;
}

bool CastParser::match(TokenType type) {
    if (peekToken(0).type == type) {
        consume();
        return true;
    }
    return false;
}

bool CastParser::check(TokenType type) {
    return peekToken(0).type == type;
}

void CastParser::expect(TokenType type, const std::string &errMsg) {
    if (peekToken(0).type != type) {
        throw std::runtime_error("Parser error on line " + std::to_string(peekToken(0).line) +
                                 ", col " + std::to_string(peekToken(0).column) + ": " + errMsg +
                                 ", got '" + peekToken(0).text + "'");
    }
    consume();
}

std::shared_ptr<ast::Program> CastParser::parseProgram() {
    auto prog = std::make_shared<ast::Program>();
    
    // Skip package, space, imports if present (ignored/not lowered in backend)
    while (peekToken(0).type != TokenType::TOK_EOF) {
        if (peekToken(0).type == TokenType::CNAME && (peekToken(0).text == "package" || peekToken(0).text == "space")) {
            consume();
            expect(TokenType::CNAME, "expected identifier");
            match(TokenType::SEMICOLON);
        } else if (peekToken(0).type == TokenType::CNAME && peekToken(0).text == "import") {
            consume();
            if (match(TokenType::LPAREN)) {
                while (peekToken(0).type != TokenType::RPAREN && peekToken(0).type != TokenType::TOK_EOF) {
                    consume();
                }
                expect(TokenType::RPAREN, "expected ')' after import list");
            } else {
                match(TokenType::LPAREN);
                expect(TokenType::CNAME, "expected import target");
                match(TokenType::RPAREN);
            }
            match(TokenType::SEMICOLON);
        } else {
            break;
        }
    }
    
    while (peekToken(0).type != TokenType::TOK_EOF) {
        auto decl = parseDecl();
        if (decl) {
            prog->decls.push_back(decl);
        } else {
            consume();
        }
    }
    return prog;
}

std::shared_ptr<ast::Decl> CastParser::parseDecl() {
    if (check(TokenType::ENUM)) {
        return parseEnumDecl();
    }
    if (check(TokenType::MACHINE)) {
        return parseMachineDecl();
    }
    if (check(TokenType::INSTANTIATE)) {
        return parseInstantiateDecl();
    }
    
    // Skip unsupported/unimplemented declarations like func, type, assertions
    if (peekToken(0).type == TokenType::CNAME) {
        std::string name = peekToken(0).text;
        if (name == "func" || name == "type" || name == "assertions") {
            consume();
            while (peekToken(0).type != TokenType::LBRACE && peekToken(0).type != TokenType::TOK_EOF) {
                consume();
            }
            if (match(TokenType::LBRACE)) {
                int depth = 1;
                while (depth > 0 && peekToken(0).type != TokenType::TOK_EOF) {
                    if (peekToken(0).type == TokenType::LBRACE) depth++;
                    else if (peekToken(0).type == TokenType::RBRACE) depth--;
                    consume();
                }
            }
            return nullptr;
        }
    }
    return nullptr;
}

std::shared_ptr<ast::EnumDecl> CastParser::parseEnumDecl() {
    consume(); // enum
    auto decl = std::make_shared<ast::EnumDecl>();
    decl->name = peekToken(0).text;
    expect(TokenType::CNAME, "expected enum name");
    expect(TokenType::EQUAL, "expected '='");
    expect(TokenType::LBRACE, "expected '{'");
    while (true) {
        decl->members.push_back(peekToken(0).text);
        expect(TokenType::CNAME, "expected enum member");
        if (!match(TokenType::COMMA)) {
            break;
        }
    }
    expect(TokenType::RBRACE, "expected '}'");
    match(TokenType::SEMICOLON);
    return decl;
}

std::shared_ptr<ast::MachineDecl> CastParser::parseMachineDecl() {
    consume(); // machine
    auto decl = std::make_shared<ast::MachineDecl>();
    decl->name = peekToken(0).text;
    expect(TokenType::CNAME, "expected machine name");
    
    // Optional compile-time parameters
    if (match(TokenType::LBRACKET)) {
        if (peekToken(0).type != TokenType::RBRACKET) {
            while (true) {
                std::string pName = peekToken(0).text;
                expect(TokenType::CNAME, "expected parameter name");
                expect(TokenType::COLON, "expected ':'");
                ast::Type pType = parseType();
                decl->compileTimeParams.push_back({pName, pType});
                if (!match(TokenType::COMMA)) {
                    break;
                }
            }
        }
        expect(TokenType::RBRACKET, "expected ']'");
    }
    
    expect(TokenType::LBRACE, "expected '{'");
    if (check(TokenType::INTERFACE)) {
        decl->interfaceBlock = parseInterfaceBlock();
    } else {
        throw std::runtime_error("expected interface block");
    }
    
    if (check(TokenType::SHARED)) {
        decl->sharedBlock = parseSharedBlock();
    }
    
    // Skip memory block if present
    if (peekToken(0).type == TokenType::CNAME && peekToken(0).text == "memory") {
        consume();
        expect(TokenType::LBRACE, "expected '{'");
        while (peekToken(0).type != TokenType::RBRACE && peekToken(0).type != TokenType::TOK_EOF) {
            consume();
        }
        expect(TokenType::RBRACE, "expected '}'");
    }
    
    if (check(TokenType::STATES)) {
        decl->statesBlock = parseStatesBlock();
    }
    
    expect(TokenType::RBRACE, "expected '}'");
    return decl;
}

std::shared_ptr<ast::InterfaceBlock> CastParser::parseInterfaceBlock() {
    consume(); // interface
    auto block = std::make_shared<ast::InterfaceBlock>();
    expect(TokenType::LBRACE, "expected '{'");
    while (check(TokenType::INPUT) || check(TokenType::OUTPUT)) {
        block->ioDecls.push_back(parseIODecl());
    }
    expect(TokenType::RBRACE, "expected '}'");
    return block;
}

ast::IODecl CastParser::parseIODecl() {
    ast::IODecl io;
    io.direction = peekToken(0).text;
    consume(); // input or output
    
    io.typedIdent.type = parseType();
    if (peekToken(0).type == TokenType::CNAME) {
        io.idents.push_back(peekToken(0).text);
        consume();
    }
    while (match(TokenType::COMMA)) {
        if (peekToken(0).type == TokenType::CNAME) {
            io.idents.push_back(peekToken(0).text);
            consume();
        }
    }
    match(TokenType::SEMICOLON);
    return io;
}

std::shared_ptr<ast::SharedBlock> CastParser::parseSharedBlock() {
    consume(); // shared
    auto block = std::make_shared<ast::SharedBlock>();
    expect(TokenType::LBRACE, "expected '{'");
    while (check(TokenType::VAR)) {
        block->varDecls.push_back(parseVarDecl());
    }
    expect(TokenType::RBRACE, "expected '}'");
    return block;
}

std::shared_ptr<ast::VarDecl> CastParser::parseVarDecl() {
    consume(); // var
    auto decl = std::make_shared<ast::VarDecl>();
    decl->typedIdent = parseIdentTyped();
    if (match(TokenType::EQUAL)) {
        decl->initExpr = parseExpr();
    }
    match(TokenType::SEMICOLON);
    return decl;
}

std::shared_ptr<ast::StatesBlock> CastParser::parseStatesBlock() {
    consume(); // states
    auto block = std::make_shared<ast::StatesBlock>();
    expect(TokenType::LBRACE, "expected '{'");
    while (peekToken(0).type == TokenType::CNAME) {
        if (peekToken(0).text == "exception") {
            block->exceptionDecls.push_back(parseExceptionDecl());
        } else {
            block->stateDecls.push_back(parseStateDecl());
        }
    }
    expect(TokenType::RBRACE, "expected '}'");
    match(TokenType::SEMICOLON);
    return block;
}

std::shared_ptr<ast::StateDecl> CastParser::parseStateDecl() {
    auto decl = std::make_shared<ast::StateDecl>();
    decl->name = peekToken(0).text;
    expect(TokenType::CNAME, "expected state name");
    expect(TokenType::COLON, "expected ':'");
    
    if (peekToken(0).type != TokenType::LBRACE) {
        while (true) {
            auto lhs = parseExpr();
            std::string op = peekToken(0).text;
            if (match(TokenType::ARROW_L) || match(TokenType::ARROW_R) || match(TokenType::EQUAL) ||
                match(TokenType::COLON_EQUAL)) {
                auto rhs = parseExpr();
                auto bin = std::make_shared<ast::BinaryStmt>();
                bin->lhs = lhs;
                bin->op = op;
                bin->rhs = rhs;
                decl->headerReceives.push_back(bin);
            } else {
                throw std::runtime_error("expected assignment in state header");
            }
            if (!match(TokenType::COMMA)) {
                break;
            }
        }
    }
    decl->body = parseBlockStmt();
    return decl;
}

std::shared_ptr<ast::ExceptionDecl> CastParser::parseExceptionDecl() {
    consume(); // exception
    auto decl = std::make_shared<ast::ExceptionDecl>();
    decl->name = peekToken(0).text;
    expect(TokenType::CNAME, "expected exception name");
    expect(TokenType::COLON, "expected ':'");
    
    if (peekToken(0).type != TokenType::LBRACE) {
        while (true) {
            auto lhs = parseExpr();
            std::string op = peekToken(0).text;
            if (match(TokenType::ARROW_L) || match(TokenType::ARROW_R) || match(TokenType::EQUAL) ||
                match(TokenType::COLON_EQUAL)) {
                auto rhs = parseExpr();
                auto bin = std::make_shared<ast::BinaryStmt>();
                bin->lhs = lhs;
                bin->op = op;
                bin->rhs = rhs;
                decl->headerReceives.push_back(bin);
            } else {
                throw std::runtime_error("expected assignment in exception header");
            }
            if (!match(TokenType::COMMA)) {
                break;
            }
        }
    }
    decl->body = parseBlockStmt();
    return decl;
}

std::shared_ptr<ast::InstantiateDecl> CastParser::parseInstantiateDecl() {
    consume(); // instantiate
    auto decl = std::make_shared<ast::InstantiateDecl>();
    decl->body = parseBlockStmt();
    return decl;
}

// ─────────────────────────────────────────────────────────────────────────────
// Statements Parsing
// ─────────────────────────────────────────────────────────────────────────────

std::shared_ptr<ast::Stmt> CastParser::parseStmt() {
    if (check(TokenType::LBRACE)) {
        return parseBlockStmt();
    }
    if (check(TokenType::IF)) {
        return parseIfStmt();
    }
    if (check(TokenType::GOTO)) {
        return parseGotoStmt();
    }
    if (check(TokenType::VAR)) {
        auto decl = parseVarDecl();
        auto stmt = std::make_shared<ast::VarDeclStmt>();
        stmt->varDecl = decl;
        return stmt;
    }
    
    // Ignored keywords (for loop, switch, return)
    if (peekToken(0).type == TokenType::CNAME && peekToken(0).text == "for") {
        return parseForStmt();
    }
    if (peekToken(0).type == TokenType::CNAME && peekToken(0).text == "switch") {
        return parseSwitchStmt();
    }
    if (peekToken(0).type == TokenType::CNAME && peekToken(0).text == "return") {
        return parseReturnStmt();
    }
    
    return parseVarDeclOrExprOrAssign();
}

std::shared_ptr<ast::BlockStmt> CastParser::parseBlockStmt() {
    auto block = std::make_shared<ast::BlockStmt>();
    expect(TokenType::LBRACE, "expected '{'");
    while (peekToken(0).type != TokenType::RBRACE && peekToken(0).type != TokenType::TOK_EOF) {
        block->statements.push_back(parseStmt());
    }
    expect(TokenType::RBRACE, "expected '}'");
    return block;
}

std::shared_ptr<ast::IfStmt> CastParser::parseIfStmt() {
    consume(); // if
    auto stmt = std::make_shared<ast::IfStmt>();
    stmt->cond = parseExpr();
    stmt->thenBranch = parseBlockStmt();
    if (match(TokenType::ELSE)) {
        if (check(TokenType::IF)) {
            stmt->elseBranch = parseIfStmt();
        } else {
            stmt->elseBranch = parseBlockStmt();
        }
    }
    return stmt;
}

std::shared_ptr<ast::GotoStmt> CastParser::parseGotoStmt() {
    consume(); // goto
    auto stmt = std::make_shared<ast::GotoStmt>();
    stmt->targetState = peekToken(0).text;
    expect(TokenType::CNAME, "expected target state name");
    match(TokenType::SEMICOLON);
    return stmt;
}

std::shared_ptr<ast::ForStmt> CastParser::parseForStmt() {
    consume(); // for
    expect(TokenType::LPAREN, "expected '('");

    auto stmt = std::make_shared<ast::ForStmt>();

    // 1. Parse init (optional)
    if (peekToken(0).type != TokenType::SEMICOLON) {
        stmt->init = parseStmt();
    } else {
        consume(); // consume ';'
    }

    // 2. Parse cond (optional)
    if (peekToken(0).type != TokenType::SEMICOLON) {
        stmt->cond = parseExpr();
    }
    expect(TokenType::SEMICOLON, "expected ';'");

    // 3. Parse update (optional)
    if (peekToken(0).type != TokenType::RPAREN) {
        stmt->update = parseForUpdate();
    }
    expect(TokenType::RPAREN, "expected ')'");

    // 4. Parse body
    stmt->body = parseBlockStmt();

    return stmt;
}

std::shared_ptr<ast::Stmt> CastParser::parseForUpdate() {
    auto lhs = parseExpr();
    std::string op = peekToken(0).text;
    if (match(TokenType::EQUAL) || match(TokenType::ARROW_L) || match(TokenType::ARROW_R) ||
        match(TokenType::COLON_EQUAL) || match(TokenType::PLUS_EQUAL) || match(TokenType::MINUS_EQUAL) ||
        match(TokenType::STAR_EQUAL) || match(TokenType::SLASH_EQUAL) || match(TokenType::XOR_EQUAL) ||
        match(TokenType::SHL_EQUAL) || match(TokenType::SHR_EQUAL)) {
        
        auto rhs = parseExpr();
        auto bin = std::make_shared<ast::BinaryStmt>();
        bin->lhs = lhs;
        bin->op = op;
        bin->rhs = rhs;
        return bin;
    }
    
    auto stmt = std::make_shared<ast::ExprStmt>();
    stmt->expr = lhs;
    return stmt;
}

std::shared_ptr<ast::SwitchStmt> CastParser::parseSwitchStmt() {
    consume(); // switch
    while (peekToken(0).type != TokenType::LBRACE && peekToken(0).type != TokenType::TOK_EOF) {
        consume();
    }
    expect(TokenType::LBRACE, "expected '{'");
    int depth = 1;
    while (depth > 0 && peekToken(0).type != TokenType::TOK_EOF) {
        if (peekToken(0).type == TokenType::LBRACE) depth++;
        else if (peekToken(0).type == TokenType::RBRACE) depth--;
        consume();
    }
    return std::make_shared<ast::SwitchStmt>();
}

std::shared_ptr<ast::ReturnStmt> CastParser::parseReturnStmt() {
    consume(); // return
    auto stmt = std::make_shared<ast::ReturnStmt>();
    if (peekToken(0).type != TokenType::SEMICOLON) {
        while (true) {
            stmt->exprs.push_back(parseExpr());
            if (!match(TokenType::COMMA)) {
                break;
            }
        }
    }
    match(TokenType::SEMICOLON);
    return stmt;
}

std::shared_ptr<ast::Stmt> CastParser::parseVarDeclOrExprOrAssign() {
    if (peekToken(0).type == TokenType::CNAME && peekToken(0).text == "const") {
        consume(); // const
        expect(TokenType::CNAME, "expected constant name");
        if (peekToken(0).type != TokenType::EQUAL) {
            parseType();
        }
        expect(TokenType::EQUAL, "expected '='");
        parseExpr();
        match(TokenType::SEMICOLON);
        return std::make_shared<ast::ExprStmt>();
    }
    
    auto lhs = parseExpr();
    std::string op = peekToken(0).text;
    if (match(TokenType::EQUAL) || match(TokenType::ARROW_L) || match(TokenType::ARROW_R) ||
        match(TokenType::COLON_EQUAL) || match(TokenType::PLUS_EQUAL) || match(TokenType::MINUS_EQUAL) ||
        match(TokenType::STAR_EQUAL) || match(TokenType::SLASH_EQUAL) || match(TokenType::XOR_EQUAL) ||
        match(TokenType::SHL_EQUAL) || match(TokenType::SHR_EQUAL)) {
        
        auto rhs = parseExpr();
        match(TokenType::SEMICOLON);
        
        if (op == "=") {
            if (auto id = std::dynamic_pointer_cast<ast::IdentExpr>(lhs)) {
                if (auto call = std::dynamic_pointer_cast<ast::CallExpr>(rhs)) {
                    auto inst = std::make_shared<ast::InstModuleStmt>();
                    inst->varName = id->name;
                    inst->callExpr = call;
                    return inst;
                }
            }
        }
        
        auto bin = std::make_shared<ast::BinaryStmt>();
        bin->lhs = lhs;
        bin->op = op;
        bin->rhs = rhs;
        return bin;
    }
    
    match(TokenType::SEMICOLON);
    auto stmt = std::make_shared<ast::ExprStmt>();
    stmt->expr = lhs;
    return stmt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Expressions Parsing
// ─────────────────────────────────────────────────────────────────────────────

std::shared_ptr<ast::Expr> CastParser::parseExpr() {
    return parseBinaryExpr(0);
}

std::shared_ptr<ast::Expr> CastParser::parseBinaryExpr(int minPrecedence) {
    auto lhs = parseUnaryExpr();
    while (true) {
        int prec = getPrecedence(peekToken(0).type);
        if (prec < minPrecedence) {
            break;
        }
        std::string op = peekToken(0).text;
        consume();
        auto rhs = parseBinaryExpr(prec + 1);
        auto bin = std::make_shared<ast::BinaryExpr>();
        bin->lhs = lhs;
        bin->op = op;
        bin->rhs = rhs;
        lhs = bin;
    }
    return lhs;
}

std::shared_ptr<ast::Expr> CastParser::parseUnaryExpr() {
    if (peekToken(0).type == TokenType::EXCL || peekToken(0).type == TokenType::AMP || peekToken(0).type == TokenType::STAR) {
        std::string op = peekToken(0).text;
        consume();
        auto un = std::make_shared<ast::UnaryExpr>();
        un->op = op;
        un->expr = parseUnaryExpr();
        return un;
    }
    return parsePostfixExpr(parsePrimaryExpr());
}

std::shared_ptr<ast::Expr> CastParser::parsePostfixExpr(std::shared_ptr<ast::Expr> base) {
    while (true) {
        if (peekToken(0).type == TokenType::PLUS_PLUS || peekToken(0).type == TokenType::MINUS_MINUS) {
            std::string op = peekToken(0).text;
            consume();
            auto upd = std::make_shared<ast::UpdateExpr>();
            upd->expr = base;
            upd->op = op;
            base = upd;
        } else if (peekToken(0).type == TokenType::LPAREN) {
            consume(); // '('
            auto call = std::make_shared<ast::CallExpr>();
            if (auto id = std::dynamic_pointer_cast<ast::IdentExpr>(base)) {
                call->funcName = id->name;
            } else if (auto field = std::dynamic_pointer_cast<ast::FieldExpr>(base)) {
                call->funcName = field->object + "." + field->field;
            } else {
                call->funcName = "";
            }
            if (peekToken(0).type != TokenType::RPAREN) {
                while (true) {
                    call->args.push_back(parseExpr());
                    if (!match(TokenType::COMMA)) {
                        break;
                    }
                }
            }
            expect(TokenType::RPAREN, "expected ')'");
            
            if (peekToken(0).type == TokenType::LBRACKET) {
                consume(); // '['
                if (peekToken(0).type != TokenType::RBRACKET) {
                    while (true) {
                        std::string paramName;
                        if (peekToken(0).type == TokenType::CNAME && peekToken(1).type == TokenType::EQUAL) {
                            paramName = peekToken(0).text;
                            consume(); // name
                            consume(); // '='
                        }
                        auto val = parseExpr();
                        call->compileTimeParams.push_back({paramName, val});
                        if (!match(TokenType::COMMA)) {
                            break;
                        }
                    }
                }
                expect(TokenType::RBRACKET, "expected ']'");
            }
            base = call;
        } else if (peekToken(0).type == TokenType::LBRACKET) {
            consume(); // '['
            while (peekToken(0).type != TokenType::RBRACKET && peekToken(0).type != TokenType::TOK_EOF) {
                consume();
            }
            expect(TokenType::RBRACKET, "expected ']'");
            auto dummy = std::make_shared<ast::NilLiteral>();
            base = dummy;
        } else if (peekToken(0).type == TokenType::DOT) {
            consume(); // '.'
            std::string fieldName = peekToken(0).text;
            expect(TokenType::CNAME, "expected field name");
            auto f = std::make_shared<ast::FieldExpr>();
            if (auto id = std::dynamic_pointer_cast<ast::IdentExpr>(base)) {
                f->object = id->name;
            }
            f->field = fieldName;
            base = f;
        } else {
            break;
        }
    }
    return base;
}

std::shared_ptr<ast::Expr> CastParser::parsePrimaryExpr() {
    if (peekToken(0).type == TokenType::INTEGER || peekToken(0).type == TokenType::HEX) {
        auto lit = std::make_shared<ast::NumberLiteral>();
        lit->text = peekToken(0).text;
        lit->value = std::stoll(peekToken(0).text, nullptr, 0);
        consume();
        return lit;
    }
    if (peekToken(0).type == TokenType::STRING_LIT) {
        auto lit = std::make_shared<ast::StringLiteral>();
        std::string raw = peekToken(0).text;
        if (raw.size() >= 2) raw = raw.substr(1, raw.size() - 2);
        lit->value = raw;
        consume();
        return lit;
    }
    if (peekToken(0).type == TokenType::NIL) {
        consume();
        return std::make_shared<ast::NilLiteral>();
    }
    if (match(TokenType::LPAREN)) {
        auto expr = parseExpr();
        expect(TokenType::RPAREN, "expected ')'");
        return expr;
    }
    if (peekToken(0).type == TokenType::CNAME) {
        std::string name = peekToken(0).text;
        consume();
        
        if (peekToken(0).type == TokenType::LBRACKET) {
            bool followedByParen = false;
            size_t i = 1;
            while (peekToken(i).type != TokenType::RBRACKET && peekToken(i).type != TokenType::TOK_EOF) {
                i++;
            }
            if (peekToken(i).type == TokenType::RBRACKET && peekToken(i + 1).type == TokenType::LPAREN) {
                followedByParen = true;
            }
            
            if (followedByParen) {
                consume(); // '['
                auto call = std::make_shared<ast::CallExpr>();
                call->funcName = name;
                if (peekToken(0).type != TokenType::RBRACKET) {
                    while (true) {
                        std::string paramName;
                        if (peekToken(0).type == TokenType::CNAME && peekToken(1).type == TokenType::EQUAL) {
                            paramName = peekToken(0).text;
                            consume(); // name
                            consume(); // '='
                        }
                        auto val = parseExpr();
                        call->compileTimeParams.push_back({paramName, val});
                        if (!match(TokenType::COMMA)) {
                            break;
                        }
                    }
                }
                expect(TokenType::RBRACKET, "expected ']'");
                expect(TokenType::LPAREN, "expected '('");
                if (peekToken(0).type != TokenType::RPAREN) {
                    while (true) {
                        call->args.push_back(parseExpr());
                        if (!match(TokenType::COMMA)) {
                            break;
                        }
                    }
                }
                expect(TokenType::RPAREN, "expected ')'");
                return call;
            }
        }
        
        auto id = std::make_shared<ast::IdentExpr>();
        id->name = name;
        return id;
    }
    throw std::runtime_error("Parser error on line " + std::to_string(peekToken(0).line) +
                             ", col " + std::to_string(peekToken(0).column) +
                             ": expected expression, got '" + peekToken(0).text + "'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Types Parsing
// ─────────────────────────────────────────────────────────────────────────────

ast::Type CastParser::parseType() {
    ast::Type t;
    if (check(TokenType::T_STRING) || check(TokenType::T_INT) || check(TokenType::T_FLOAT) ||
        check(TokenType::T_BYTE) || check(TokenType::T_INT32) || check(TokenType::T_UINT32) ||
        check(TokenType::T_UINT16) || check(TokenType::T_BOOL)) {
        t.kind = ast::TypeKind::PRIMITIVE;
        t.name = peekToken(0).text;
        consume();
    } else if (match(TokenType::LBRACKET)) {
        if (peekToken(0).type == TokenType::INTEGER) {
            t.kind = ast::TypeKind::ARRAY;
            t.size = std::stoll(peekToken(0).text);
            consume();
            expect(TokenType::RBRACKET, "expected ']'");
            t.elementType = std::make_shared<ast::Type>(parseType());
        } else {
            t.kind = ast::TypeKind::SLICE;
            expect(TokenType::RBRACKET, "expected ']'");
            t.elementType = std::make_shared<ast::Type>(parseType());
        }
    } else if (peekToken(0).type == TokenType::CNAME && peekToken(0).text == "chan") {
        t.kind = ast::TypeKind::CHAN;
        consume();
        if (match(TokenType::ARROW_L)) {
            t.chanDir = "<-";
        } else if (match(TokenType::ARROW_R)) {
            t.chanDir = "->";
        }
        t.elementType = std::make_shared<ast::Type>(parseType());
    } else if (peekToken(0).type == TokenType::CNAME && peekToken(1).type == TokenType::CNAME) {
        t.kind = ast::TypeKind::CUSTOM;
        t.name = peekToken(0).text;
        consume();
    } else {
        t.kind = ast::TypeKind::CUSTOM;
        t.name = "";
    }
    return t;
}

ast::IdentTyped CastParser::parseIdentTyped() {
    ast::IdentTyped it;
    it.type = parseType();
    
    if (it.type.kind == ast::TypeKind::CUSTOM && it.type.name.empty()) {
        if (peekToken(0).type == TokenType::CNAME) {
            it.idents.push_back(peekToken(0).text);
            consume();
        }
    } else {
        if (peekToken(0).type == TokenType::CNAME) {
            it.idents.push_back(peekToken(0).text);
            consume();
        }
    }
    
    while (match(TokenType::COMMA)) {
        if (peekToken(0).type == TokenType::CNAME) {
            it.idents.push_back(peekToken(0).text);
            consume();
        }
    }
    return it;
}
