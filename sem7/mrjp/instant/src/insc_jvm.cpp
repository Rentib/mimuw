#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stack>
#include <vector>

#include "util.h"

size_t height(Exp start)
{
	static map<Exp, size_t> h;

	struct Frame {
		Exp e;
		bool done;
	};

	if (!start) return 0;
	if (h.contains(start)) return h[start];

	stack<Frame> todo;
	todo.emplace(start, false);

	while (!todo.empty()) {
		auto [e, done] = todo.top();
		todo.pop();

		if (!e) continue;
		if (e->kind == Exp_::is_ExpLit || e->kind == Exp_::is_ExpVar) {
			h[e] = 1;
			continue;
		}

		Exp lhs = get_lhs(e), rhs = get_rhs(e);
		if (done) {
			size_t lh = height(lhs), rh = height(rhs);
			h[e] = min(max(lh, rh + 1), max(lh + 1, rh));
		} else {
			todo.emplace(e, true);
			todo.emplace(rhs, false);
			todo.emplace(lhs, false);
		}
	}

	return h[start];
}

string compile(Exp e, const map<string, int> &state)
{
	string res = "";

	struct Frame {
		Exp e;
		bool done;
		bool swap;
	};

	stack<Frame> todo;
	todo.emplace(e, false, false);

	while (!todo.empty()) {
		auto [e, done, swap] = todo.top();
		todo.pop();

		if (!e) continue;
		if (e->kind == Exp_::is_ExpLit) {
			int n = e->u.explit_.integer_;
			if (n <= 5)
				res += "\ticonst_";
			else if (n <= INT8_MAX)
				res += "\tbipush ";
			else if (n <= INT16_MAX)
				res += "\tsipush ";
			else
				res += "\tldc ";
			res += to_string(n) + "\n";
			continue;
		} else if (e->kind == Exp_::is_ExpVar) {
			if (!state.contains(e->u.expvar_.ident_))
				die("undeclared variable ",
				    e->u.expvar_.ident_);
			int n = state.at(e->u.expvar_.ident_);
			res += "\tiload";
			res += (n <= 3 ? "_" : " ") + to_string(n);
			res += string(" ; ") + e->u.expvar_.ident_ + "\n";
			continue;
		}

		if (done) {
			if (swap && (e->kind == Exp_::is_ExpSub ||
				     e->kind == Exp_::is_ExpDiv))
				res += "\tswap\n";

			if (e->kind == Exp_::is_ExpAdd) res += "\tiadd\n";
			if (e->kind == Exp_::is_ExpSub) res += "\tisub\n";
			if (e->kind == Exp_::is_ExpMul) res += "\timul\n";
			if (e->kind == Exp_::is_ExpDiv) res += "\tidiv\n";
			continue;
		}

		Exp lhs = get_lhs(e), rhs = get_rhs(e);
		vector<Frame> spush;

		if (height(lhs) < height(rhs)) {
			swap = true;
			std::swap(lhs, rhs);
		}

		todo.emplace(e, true, swap);
		todo.emplace(rhs, false, false);
		todo.emplace(lhs, false, false);
	}

	return res;
}

static size_t stack_height = 0;

string compile(Stmt s, map<string, int> &state)
{
	string res = "";

	Exp exp;
	char *ident;
	int n;

	if (s->kind == Stmt_::is_SExp) goto SExp;

	exp = s->u.sass_.exp_;
	ident = s->u.sass_.ident_;
	n = state.contains(ident) ? state[ident]
				  : (state[ident] = state.size());
	res += compile(exp, state);
	res += "\tistore";
	res += (n <= 3 ? "_" : " ");
	res += to_string(n) + " ; " + ident + "\n";

	stack_height = max(stack_height, height(exp));
	return res;

SExp:
	exp = s->u.sexp_.exp_;

	res += compile(exp, state);
	res += "\tgetstatic java/lang/System/out Ljava/io/PrintStream;\n";
	res += "\tswap\n";
	res += "\tinvokevirtual java/io/PrintStream/println(I)V\n";

	stack_height = max(stack_height, height(exp) + 1);
	return res;
}

string compile(Program ast, const string &name)
{
	string res = "";

	map<string, int> state;
	vector<string> code;

	for (ListStmt l = ast->u.prog_.liststmt_; l; l = l->liststmt_)
		code.emplace_back(compile(l->stmt_, state));
	string bytecode = accumulate(code.begin(), code.end(), string(""));

	size_t stack = stack_height, locals = state.size() + 1 /* this */;

	res += ".class public " + name + "\n";
	res += ".super java/lang/Object\n";
	res += "\n";
	res += ".method public <init>()V\n";
	res += "\taload_0\n";
	res += "\tinvokespecial java/lang/Object/<init>()V\n";
	res += "\treturn\n";
	res += ".end method\n";
	res += "\n";
	res += ".method public static main([Ljava/lang/String;)V\n";
	res += ".limit stack " + to_string(stack) + "\n";
	res += ".limit locals " + to_string(locals) + "\n";
	res += bytecode;
	res += "\treturn\n";
	res += ".end method\n";

	return res;
}

int main(int argc, char *argv[])
{
	string name;
	Program ast;

	if (argc > 1) {
		name = argv[1];

		FILE *in = fopen(argv[1], "r");
		if (!in) die("cannot open file ", name);

		ast = pProgram(in);
		if (fclose(in) != 0) die("cannot close file ", name);
	} else {
		ast = pProgram(stdin);
		name = "out.ins";
	}

	if (!ast) die("parse error, see bnfc(1)");

	string bytecode = compile(ast, filesystem::path(name).stem());
	string jname = filesystem::path(name).replace_extension("j");
	ofstream out(jname);
	if (!out) die("cannot open file ", jname);
	out << bytecode;
	out.close();

	auto parent_path = filesystem::path(name).parent_path().string();
	auto dir = parent_path.empty() ? "." : parent_path;

	if (filesystem::exists("./lib/jasmin.jar")) {
		string cmd = "java -jar ./lib/jasmin.jar " + jname + " -d " + dir;
		if (system(cmd.c_str()) != 0) die("cannot run jasmin");
	} else {
		warn("jasmin.jar not found, skipping assembly");
	}

	free_Program(ast);
	return 0;
}
