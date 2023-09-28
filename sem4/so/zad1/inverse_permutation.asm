	section	.text
BIT	equ	1 << 31
	; Arguments of the function are a pointer p to a non-empty array of
	; integers and the size of this array n. If the array pointed to by
	; p contains a permutation of numbers from the range from 0 to n-1,
	; then the function inverts this permutation in place and the result of
	; the function is true. Otherwise, the result of the function is false,
	; and the content of the array pointed to by p after the function is
	; executed is the same as at the time of its call.
	global	inverse_permutation
inverse_permutation:				; rdi - n, rsi - p
	test	rdi, rdi
	jz	.fail				; n == 0
	mov	rax, 0x80000000
	cmp	rdi, rax
	ja	.fail				; n > INT_MAX + 1

	xor	ecx, ecx			; i := 0
.check_args:
	mov	edx, dword [rsi + 4*rcx]	; j := p[i]
	and	edx, ~BIT			; make sure j is not marked
	cmp	rdx, rdi			; check if j is greater or equal to n
	jae	.out_of_range_or_duplicate_element
	test	byte [rsi + 4*rdx + 3], 128	; check if p[j] is less than 0 or has been already marked
	jnz	.out_of_range_or_duplicate_element
	or	byte [rsi + 4*rdx + 3], 128	; mark p[j]
	inc	rcx
	cmp	rdi, rcx
	jne	.check_args

	; NOTE: the above loop has finished so rcx == n
.for:
	lea	r9, [rcx - 1]			; i := rcx - 1
	mov	eax, dword [rsi + 4*r9]		; cur := p[i]
	xor	eax, BIT			; make sure its unmarked
	js	.rof				; skip if it has already been visited
	mov	edx, r9d			; prv := i
.while:
	cmp	r9d, eax			; compare i with cur
	je	.elihw				; break after full cycle
	mov	r8d, dword [rsi + 4*rax]	; nxt := p[cur]
	mov	dword [rsi + 4*rax], edx	; p[cur] := prv (inverse permutation)
	mov	edx, eax			; prv := cur
	mov	eax, r8d			; cur := nxt
	and	eax, ~BIT			; unmark it
	jmp	.while
.elihw:
	mov	dword [rsi + 4*r9], edx		; p[i] := prv (inverse permutation)
.rof:
	loop	.for

	mov	al, 1
	ret					; return true

.out_of_range_or_duplicate_element:
	test	ecx, ecx
	jz	.fail				; dont go into the loop if i == 0
.unmark:
	mov	eax, dword [rsi + 4*rcx - 4]	; j := p[i]
	and	eax, ~BIT			; make sure j is unmarked
	and	byte [rsi + 4*rax + 3], 127	; unmark p[j]
	loop	.unmark
.fail:
	xor	eax, eax
	ret					; return false
