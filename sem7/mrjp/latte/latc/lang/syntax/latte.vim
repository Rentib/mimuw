" Vim syntax file
" Language: latte
" Add this to your vimrc "au bufreadpre,bufnewfile *.latte set ft=latte" 
" and put this file in ~/.vim/syntax/
" Author: Stanisław Bitner <https://github.com/Rentib>

if exists("b:current_syntax")
    finish
endif

syn case match

syn region latteComment start="/\*" end="\*/" contains=latteTodo
syn match latteComment "//.*$" contains=latteTodo
syn match latteComment "^\s*#.*$" contains=latteTodo

syn region latteString		start=+"+  skip=+\\\\\|\\"+  end=+"+
syn region latteString		start=+'+  skip=+\\\\\|\\'+  end=+'+
syn region latteString		start=+`+  skip=+\\\\\|\\`+  end=+`+

syn match   latteNumber "[-+]\=\(\<\d[[:digit:]_]*L\=\>\|0[xX]\x[[:xdigit:]_]*\>\)"
syn keyword latteBoolean true false

syn match latteFunction "\<[a-zA-Z][a-zA-Z0-9_]*\s*("me=e-1

syn match latteOperator "+"
syn match latteOperator "-"
syn match latteOperator "*"
syn match latteOperator "\/\(.*[\/\*]\)\@!"
syn match latteOperator "%"
syn match latteOperator "!"
syn match latteOperator "&&"
syn match latteOperator "||"
syn match latteOperator ">"
syn match latteOperator ">="
syn match latteOperator "<"
syn match latteOperator "<="
syn match latteOperator "=="
syn match latteOperator "!="
syn match latteOperator "="

syn keyword latteType        int string boolean void
syn keyword latteStatement   return
syn keyword latteConditional if else
syn keyword latteRepeat      while
syn keyword latteKeyword     return class
syn keyword latteTypedef     self
syn match   latteTypedef     "\.\s*\<class\>"ms=s+1
syn match   latteClassDecl	 "^class\>"
syn match   latteClassDecl	 "[^.]\s*\<class\>"ms=s+1
syn keyword latteClassDecl   extends
syn keyword latteOperator    new

syn match latteIgnore ";"
syn match latteIgnore ","
syn match latteIgnore "\s"

hi def link latteComment        Comment
hi def link latteString         String
hi def link latteNumber         Number
hi def link latteBoolean        Boolean
hi def link latteConditional    Conditional
hi def link latteRepeat         Repeat
hi def link latteOperator       Operator
hi def link latteKeyword        Statement
hi def link latteType           Type
hi def link latteTypedef        Typedef
hi def link latteClassDecl      StorageClass
hi def link latteFunction       Function
hi def link latteIgnore         Ignore
hi def link latteTodo           Todo

let b:current_syntax = "latte"
