%assign	BEGIN		0x7C00
%assign	SECTOR_SIZE	512

%assign	VIDEO_SERVICES		0x10
%assign	DISK_SERVICES		0x13
%assign	KEYBOARD_SERVICES	0x16
%assign	RTC_SERVICES		0x1A	; Real Time Clock

[bits 16]
[org BEGIN]

jmp	0:start				; first 3 bytes
nop

start:
	xor	ax, ax			; clear segment registers
	mov	ds, ax
	mov	es, ax
	mov	ss, ax			; setup stack just below the bootloader
	mov	sp, BEGIN

	mov	ax, 0x0203		; read 3 sectors
	xor	dh, dh			; head
					; dl is drive
	xor	ch, ch			; cylinder
	mov	cl, 2			; first sector
	xor	bx, bx
	mov	es, bx
	mov	bx, seg2
	int	DISK_SERVICES

	jc	halt			; disk error
	cmp	al, 3
	jne	halt			; not all sectors loaded

	jmp	0:main

;-----------------------------------------------------------------------------
; Halts program execution
;-----------------------------------------------------------------------------
halt:
	cli				; clear interrupts
	hlt				; halt CPU

;-----------------------------------------------------------------------------
; Prints AX as an unsigned integer
;-----------------------------------------------------------------------------
print_int:
	pusha

	mov	cx, 5			; at most 5 digits
	mov	bx, 10
.loop:
	xor	dx, dx
	div	bx
	push	ax

	mov	ah, 0x0A		; write character in TTY mode
	mov	al, dl			; remainder
	add	al, '0'			; change to ascii
	int	VIDEO_SERVICES

	pop	ax
	loop	.loop

	popa
	ret

;-----------------------------------------------------------------------------
; Prints string at address SI up to EOL
;-----------------------------------------------------------------------------
print_string:
	pusha

.while:
	mov	al, byte [si]
	mov	ah, 0x0E		; write character in TTY mode
	int	VIDEO_SERVICES

	inc	si			; next char
	cmp	al, `\n`		; check if EOL
	jne	.while

	popa
	ret

;-----------------------------------------------------------------------------
; Set cursor position at row DH and column DL
;-----------------------------------------------------------------------------
set_cursor:
	mov	ah, 0x02		; set cursor position
	xor	bh, bh			; 0th page
	int	VIDEO_SERVICES
	ret

times	SECTOR_SIZE-2-($-$$) db 0	; pad to 1 sector
dw	0xAA55				; boot signature

;-----------------------------------------------------------------------------
; Start of purpose
;-----------------------------------------------------------------------------
seg2:

timer:
.current:	dw 0x0
.best:		dw 0xFFFF

main:
	mov	ax, 0x0003		; set video mode to text (80x25)
	int	VIDEO_SERVICES

.remember_time:
	; we don't use RTC in any other way than measuring time of
	; the loop, so we can clear it instead of remembering
	; this also helps avoid problem with using the program around midnight
	; where it would be necessary to check if day has passed since
	; start of loop
	mov	ah, 0x01		; set system clock counter
	xor	cx, cx			; high order word of tick count
	xor	dx, dx			; low order word of tick count
	int	RTC_SERVICES

.clear_first_7_rows:
	mov	ax, 0x0600		; clear screen
	xor	cx, cx			; upper left row and column
	mov	dx, 0x064F		; upper left row and column
	mov	bh, 0x07		; white fg, black bg
	int	VIDEO_SERVICES

	xor	dx, dx			; we start at row 0, column 0
	mov	si, string		; address of string
	call	set_cursor

.print_new_row:
	call	set_cursor
	call	print_string
.loop:
	call	set_cursor
	xor	ah, ah			; read character
	int	KEYBOARD_SERVICES

	cmp	al, 0x1B		; check if Escape has been pressed
	je	.remember_time

	cmp	al, byte [si]		; check if correct key was pressed
	je	.ok_key

.err_key:
	movzx	ax, dl
	sub	si, ax			; move string pointer back
	xor	dl, dl			; set column to 0
	jmp	.loop
.ok_key:
	inc	si			; increase string pointer
	inc	dl			; increase column
	cmp	al, `\r`		; check EOL
	jne	.loop			; if not, continue
	inc	si			; increase string pointer again to skip `\n`
	inc	dh			; increase row
	xor	dl, dl			; set column to 0

	cmp	byte [si], `\0`		; check null byte
	jne	.print_new_row		; if not, continue

.show_time:
	pusha

	xor	ah, ah			; read system clock counter
	int	RTC_SERVICES

	mov	bx, 18			; ~18 ticks per second
	mov	ax, dx			; low order word
	xor	dx, dx			; setup for division
	div	bx
	xchg	ax, cx			; high order word
	mov	bx, 1<<16 / 18		; change to seconds
	mul	bx			; 2^16 / 18 (seconds in higher word)

	add	ax, cx			; hope for no overflow

	mov	word [timer.current], ax

	mov	bx, word [timer.best]
	cmp	ax, bx
	cmova	ax, bx
	mov	word [timer.best], ax	; best = min(best, current)

	mov	dx, 0x1800		; bottom left corner
	call	set_cursor
	mov	ax, word [timer.current]
	call	print_int		; print current time

	mov	dl, 80-5		; bottom right corner
	call	set_cursor
	mov	ax, word [timer.best]
	call	print_int		; print best time

	popa
.wait_for_escape:
	call	set_cursor

	xor	ah, ah			; read character
	int	KEYBOARD_SERVICES

	cmp	al, 0x1B		; check if Escape has been pressed
	je	.remember_time
	jmp	.wait_for_escape

string:	db `1\r\n2\r\n3\r\n4\r\n5\r\n6\r\n7\r\n`, 0
length:	equ $-string

times	4*SECTOR_SIZE-($-$$) db 0	; pad to 4 sectors 
