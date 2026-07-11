#include <iostream>
#include <fstream>
#include <string>
#include <streambuf>
#include "CastParser.hpp"
#include "CastAST.hpp"
#include "castLower.hpp"
#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/SeqToSV.h>
#include <circt/Dialect/ESI/ESIPasses.h>
#include <circt/Dialect/Seq/SeqPasses.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Conversion/VerifToSV.h>
#include <mlir/Pass/PassManager.h>

static void printUsage(const char *prog)
{
    std::cerr << "Usage: " << prog
              << " [--tb [--duration=<ns>] [--vcd=<path>]] <input.cast>\n"
              << "  --tb              embed a simulation testbench in the output\n"
              << "  --duration=<ns>   simulation run time in ns (default: 500)\n"
              << "  --vcd=<path>      VCD dump path (default: /tmp/<basename>.vcd)\n";
}

int main(int argc, const char *argv[])
{
    // ── argument parsing ──────────────────────────────────────────────────────
    bool emitTb = false;
    int  simDuration = 500;
    std::string vcdPath;
    std::string inputFile;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--tb")
        {
            emitTb = true;
        }
        else if (arg.starts_with("--duration="))
        {
            simDuration = std::stoi(arg.substr(11));
        }
        else if (arg.starts_with("--vcd="))
        {
            vcdPath = arg.substr(6);
        }
        else if (arg.starts_with("--"))
        {
            std::cerr << "error: unknown flag '" << arg << "'\n";
            printUsage(argv[0]);
            return 1;
        }
        else
        {
            if (!inputFile.empty())
            {
                std::cerr << "error: multiple input files specified\n";
                printUsage(argv[0]);
                return 1;
            }
            inputFile = arg;
        }
    }

    if (inputFile.empty())
    {
        printUsage(argv[0]);
        return 1;
    }

    // Derive VCD path from input basename if not set.
    if (vcdPath.empty())
    {
        std::string base = inputFile;
        auto slash = base.rfind('/');
        if (slash != std::string::npos) base = base.substr(slash + 1);
        auto dot = base.rfind('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        vcdPath = "/tmp/" + base + ".vcd";
    }

    // ── parse ─────────────────────────────────────────────────────────────────
    std::ifstream stream(inputFile);
    if (!stream.is_open())
    {
        std::cerr << "error: could not open file " << inputFile << "\n";
        return 1;
    }

    std::string sourceStr((std::istreambuf_iterator<char>(stream)),
                          std::istreambuf_iterator<char>());

    CastParser parser(sourceStr);
    std::shared_ptr<ast::Program> program;
    try
    {
        program = parser.parseProgram();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Compilation failed: " << e.what() << "\n";
        return 1;
    }

    LowerVisitor visitor;
    visitor.visit(program);

    // ── lower to SV ───────────────────────────────────────────────────────────
    mlir::PassManager pm(visitor.topModule->getContext());
    pm.addPass(circt::esi::createESIPhysicalLoweringPass());
    pm.addPass(circt::esi::createESIBundleLoweringPass());
    pm.addPass(circt::esi::createESIPortLoweringPass());
    pm.addPass(circt::esi::createESITypeLoweringPass());
    pm.addPass(circt::esi::createESItoHWPass());
    pm.nest<circt::hw::HWModuleOp>().addPass(circt::seq::createLowerSeqFIFO());
    pm.nest<circt::hw::HWModuleOp>().addPass(circt::seq::createLowerSeqHLMem());
    pm.nest<circt::hw::HWModuleOp>().addPass(circt::createLowerVerifToSVPass());
    pm.addPass(circt::createLowerSeqToSVPass());
    if (mlir::failed(pm.run(visitor.topModule)))
    {
        std::cerr << "error: lowering pipeline failed\n";
        return 1;
    }

    // Prepend timescale when embedding a testbench.
    if (emitTb)
        llvm::outs() << "`timescale 1ns/1ps\n";

    if (mlir::failed(circt::exportVerilog(visitor.topModule, llvm::outs())))
    {
        std::cerr << "error: Verilog export failed\n";
        return 1;
    }

    // ── optional testbench ────────────────────────────────────────────────────
    if (emitTb)
    {
        llvm::outs()
            << "\nmodule tb;\n"
            << "  reg clk = 0;\n"
            << "  reg rst = 1;\n"
            << "\n"
            << "  always #5 clk = ~clk;\n"
            << "\n"
            << "  Main dut (\n"
            << "    .clk (clk),\n"
            << "    .rst (rst)\n"
            << "  );\n"
            << "\n"
            << "  initial begin\n"
            << "    @(posedge clk); #1;\n"
            << "    rst = 0;\n"
            << "  end\n"
            << "\n"
            << "  initial begin\n"
            << "    $dumpfile(\"" << vcdPath << "\");\n"
            << "    $dumpvars(0, tb);\n"
            << "    #" << simDuration << ";\n"
            << "    $finish;\n"
            << "  end\n"
            << "endmodule\n";
    }

    return 0;
}
