#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stack>

#include "util.h"

class State
{
      public:
	State() : intermidiate(1) {}

	string allocate(string ident)
	{
		if (vars.contains(ident)) return "";
		vars[ident] = vars.size();
		return "\t%v" + to_string(vars[ident]) +
		       " = alloca i32, align 4\n";
	}

	string var(string ident) const
	{
		if (!vars.contains(ident)) die("undeclared variable ", ident);
		return "%v" + to_string(vars.at(ident));
	}

	string next() { return "%" + to_string(intermidiate++); }
	string curr()
	{
		if (intermidiate == 0) die("no intermidiate variables");
		return "%" + to_string(intermidiate - 1);
	}

      private:
	map<string, size_t> vars;
	size_t intermidiate;
};

string compile(Exp start, State &state)
{
	string res = "";

	map<Exp, string> t;

	stack<pair<Exp, bool>> todo;
	todo.emplace(start, false);

	while (!todo.empty()) {
		auto [e, done] = todo.top();
		todo.pop();

		if (!e) continue;
		if (e->kind == Exp_::is_ExpLit) {
			int n = e->u.explit_.integer_;
			t[e] = state.next();
			res += "\t" + t[e] + " = add i32 0, " + to_string(n) +
			       "\n";
			continue;
		} else if (e->kind == Exp_::is_ExpVar) {
			auto ident = e->u.expvar_.ident_;
			t[e] = state.next();
			res += "\t" + t[e] + " = load i32, i32* " +
			       state.var(ident) + ", align 4, !tbaa !1\n";
			continue;
		}

		auto lhs = get_lhs(e), rhs = get_rhs(e);
		if (done) {
			auto op = e->kind == Exp_::is_ExpAdd ? "add"
				: e->kind == Exp_::is_ExpSub ? "sub"
				: e->kind == Exp_::is_ExpMul ? "mul"
				: e->kind == Exp_::is_ExpDiv ? "sdiv"
							     : "";
			t[e] = state.next();
			res += "\t" + t[e] + " = " + op + " i32 " + t[lhs] +
			       ", " + t[rhs] + "\n";
		} else {
			todo.emplace(e, true);
			todo.emplace(rhs, false);
			todo.emplace(lhs, false);
		}
	}

	return res;
}

string compile(Stmt s, State &state)
{
	string res = "";

	if (s->kind == Stmt_::is_SAss) {
		auto exp = s->u.sass_.exp_;
		auto ident = s->u.sass_.ident_;

		res += compile(exp, state);
		res += state.allocate(ident);
		res += "\tstore i32 " + state.curr() + ", i32* " +
		       state.var(ident) + "\n";
	} else if (s->kind == Stmt_::is_SExp) {
		auto exp = s->u.sexp_.exp_;
		res += compile(exp, state);
		res += "\ttail call void @printInt(i32 noundef " +
		       state.curr() + ")\n";
	}

	return res;
}

string compile(Program ast, const string &name)
{
	string res = "";

	res += "source_filename = \"" + name + "\"\n\n";

	res += "@.str = private unnamed_addr constant [4 x i8] "
	       "c\"%d\\0A\\00\", align 1\n\n";
	res += "define dso_local void @printInt(i32 noundef %0) "
	       "local_unnamed_addr #0 {\n";
	res += "\ttail call i32 (i8*, ...) @printf(i8* noundef nonnull "
	       "dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* "
	       "@.str, i64 0, i64 0), i32 noundef %0)\n";
	res += "\tret void\n";
	res += "}\n\n";

	res += "declare noundef i32 @printf(i8* nocapture noundef readonly, "
	       "...) local_unnamed_addr #1\n\n";

	res += "define dso_local noundef i32 @main() local_unnamed_addr #0 {\n";

	State state;
	for (ListStmt l = ast->u.prog_.liststmt_; l; l = l->liststmt_)
		res += compile(l->stmt_, state);

	res += "\tret i32 0\n";
	res += "}\n\n";

	res += "!llvm.module.flags = !{!0}\n\n";

	res += "!0 = !{i32 7, !\"uwtable\", i32 2}\n";
	res += "!1 = !{!2, !2, i64 0}\n";
	res += "!2 = !{!\"int\", !3, i64 0}\n";
	res += "!3 = !{!\"omnipotent char\", !4, i64 0}\n";
	res += "!4 = !{!\"Simple C/C++ TBAA\"}\n";

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

	string bytecode = compile(ast, name);
	string llname = filesystem::path(name).replace_extension("ll");
	string bcname = filesystem::path(name).replace_extension("bc");
	ofstream out(llname);
	if (!out) die("cannot open file ", llname);
	out << bytecode;
	out.close();

	string cmd = "llvm-as " + llname + " -o " + bcname;
	if (system(cmd.c_str()) != 0) die("cannot run llvm-as");

	free_Program(ast);
	return 0;
}
