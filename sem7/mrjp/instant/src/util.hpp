extern "C" {
#include "Absyn.h"
#include "Parser.h"
};

using namespace std;

template <typename... Args> [[noreturn]] inline void die(Args &&...args)
{
	static std::ios_base::Init init;
	((cerr << "insc: error: ") << ... << std::forward<Args>(args)) << endl;
	exit(1);
}

template <typename... Args> inline void warn(Args &&...args)
{
	static std::ios_base::Init init;
	((cerr << "insc: warning: ") << ... << std::forward<Args>(args))
	    << endl;
}

auto get_lhs(Exp e)
{
	return e->kind == Exp_::is_ExpAdd ? e->u.expadd_.exp_1
	     : e->kind == Exp_::is_ExpSub ? e->u.expsub_.exp_1
	     : e->kind == Exp_::is_ExpMul ? e->u.expmul_.exp_1
	     : e->kind == Exp_::is_ExpDiv ? e->u.expdiv_.exp_1
					  : nullptr;
}

auto get_rhs(Exp e)
{
	return e->kind == Exp_::is_ExpAdd ? e->u.expadd_.exp_2
	     : e->kind == Exp_::is_ExpSub ? e->u.expsub_.exp_2
	     : e->kind == Exp_::is_ExpMul ? e->u.expmul_.exp_2
	     : e->kind == Exp_::is_ExpDiv ? e->u.expdiv_.exp_2
					  : nullptr;
}

