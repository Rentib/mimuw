; vim: filetype=nasm
; NOTE:
; to get nice looking code in vim, use the following command:
; :set tabstop=8 shiftwidth=8 softtabstop=8 noexpandtab
; :set comments=":;" commentstring="; %s"

global	core: function	; (uint64_t n, const char *p) (rdi, rsi)
extern	get_value	; (uint64_t n) (rdi)
extern	put_value	; (uint64_t n, uint64_t v) (rdi, rsi)

SECTION .macro

%macro save_preserved 0
	push	rbx
	push	rbp
%endmacro

%macro restore_preserved 0
	pop	rbp
	pop	rbx
%endmacro

SECTION	.text	align=1 exec

core:					; core function
					; (uint64_t n, const char *p) (rdi, rsi)
					; for more details see problem statement

	save_preserved			; store modified preserved registers
	mov	rbp, rsp		; rbp := stack pointer to later restore it
	mov	rbx, rsi		; rbx := p
	jmp	.for

; NOTE: labels are arranged in a way that makes all jumps short jumps

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.S:
	pop	rax			; m := top value (core number we wait for)
	lea	rcx, [rel waiting]	; rcx := waiting array
	lea	rdx, [rel value]	; rdx := value array
	pop	qword[rdx + rdi*8]	; value[n] := second top value (value we want to swap)
	mov	qword[rcx + rdi*8], rax ; waiting[n] := m

.lock1:					; wait until core m wants to swap with core n
	cmp	qword[rcx + rax*8], rdi	; check if waiting[m] == n
	jne	.lock1			; if not, wait

	push	qword[rdx + rax*8]	; push value[m] to the stack (swap is completed)
	xchg	qword[rcx + rax*8], rax	; waiting[m] := m (core m is not waiting for anything)

.lock2:					; wait for core m to finish
	cmp	qword[rcx + rdi*8], rdi	; check if waiting[n] == n
	jne	.lock2			; if not, wait
	jmp	.for
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.P:
	lea	rdx, [rel put_value]	; store function address of put_value in rdx
	pop	rsi			; rsi := value (abi - v argument of put_value)
	jmp	.aligned_call		; jump to aligned call
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.G:
	lea	rdx, [rel get_value]	; store function address of get_value in rdx
	stc				; set carry flag to indicate that we want to push result of the function
					; don't jump to aligned_call as it is right below
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.aligned_call:
	pushf				; save flags
	push	rdi			; save n on stack (abi - n argument of get_value and put_value)

	mov	rax, rsp		; rax := rsp
	and	rsp, ~15		; align rsp
	times	2 push	rax		; push rax twice to keep the alignment
	call	rdx			; call function stored in rdx
	pop	rsp			; restore rsp

	pop	rdi			; restore n
	popf				; restore flags

	jnc	.for			; if c flag is set than we want to push result (rax), else we don't want to
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.push_rax:				; a lot of labels would end with push rax, we can reduce size of object file by doing it once
	push	rax
.for:
	movzx	eax, byte[rbx]		; eax := *p (zero extend to get rid of trash in the register)
	test	al, al			; finish if *p == 0 (null terminated strings end at this point)
	jnz	.options
.end:
	pop	rax			; pop the result
	mov	rsp, rbp		; restore stack pointer (note that rsp was stored in rbp at the beginning)
	restore_preserved		; restore modified preserved registers
	ret

.options:				; go to correct option (shorter and faster than a jump table)
	inc	rbx
	cmp	al, 'S'
	je	.S
	ja	.n			; only n has higher ASCII code than S
	cmp	al, 'G'
	je	.G
	ja	.P			; now only P has higher ASCII code than G
	cmp	al, 'D'
	je	.D
	ja	.E			; now only E has higher ASCII code than D
	cmp	al, 'B'
	je	.B
	ja	.C			; now only C has higher ASCII code than B
	cmp	al, '-'
	je	.neg
	ja	.digit			; now only digits have higher ASCII codes than -
	cmp	al, '*'
	je	.mul
	; nothing matched so *p = '+' (.add is first so no need for jump)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.add:
	pop	rax
	add	qword[rsp], rax

	jmp	.for
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.mul:
	pop	rax
	pop	rdx
	mul	rdx

	jmp	.push_rax		; push result to the stack
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.neg:
	neg	qword[rsp]
	jmp	.for
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.digit:
	sub	al, '0'			; offset to get the correct value of a digit
	jmp	.push_rax
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.n:
	push	rdi			; push n to the stack
	jmp	.for
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.B:
	pop	rax

	cmp	qword[rsp], 0		; check if top of stack is 0
	jz	.for

	add	rbx, rax		; jump to the correct position in the string

	jmp	.for
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.C:
	pop	rdx			; pop and discard (faster than add rsp, 8)
	jmp	.for
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.D:
	pop	rax			; get top value
	push	rax			; restore top value
	jmp	.push_rax		; duplicate it
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
.E:
	pop	rdx			; get top value
	pop	rax			; get second top value
	push	rdx			; push rax first to change order
	jmp	.push_rax		; push rdx second to change order

SECTION .data	align=16 noexec

waiting:				; waiting[n] = m (number of core that n is waiting for or n/N if it is not waiting)
	times	N dq N

SECTION	.bss	align=16 noexec

value:					; value[n] = v (value of core n used in core.S, helps with synchronization)
	resq	N
