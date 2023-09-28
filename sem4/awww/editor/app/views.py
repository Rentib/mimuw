import os
import subprocess
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

from .models import Directory, File

# Create your views here.

@login_required
def index(request):
    return render(request, 'app/index.html')

@login_required
def app(request):
    return redirect('index')

@login_required
def load_tree(request):
    if request.method != 'GET': return JsonResponse({'success': False})
    root = Directory.find_root(request.user)
    return JsonResponse({'success': True, 'tree': root.print()})

@login_required
def load_file(request):
    if request.method != 'GET': return JsonResponse({'success': False})
    file_id = request.session.get('file_id')
    if file_id and File.objects.filter(id=file_id).exists() and File.objects.get(id=file_id).available:
        return JsonResponse({'success': True, 'content': File.objects.get(id=file_id).content})
    return JsonResponse({'success': False})

@login_required
def add_child(request):
    if request.method == 'POST':
        parent_id = request.POST['parent_id']
        parent = Directory.objects.get(id=parent_id)
        name = request.POST['name']
        type = request.POST['type']
        user = request.user
        if parent and name and type:
            child = None
            if type == 'file':
                child = File(name=name, owner=user, parent=parent)
            elif type == 'directory':
                child = Directory(name=name, owner=user, parent=parent)
            if child: child.save()
        return JsonResponse({'success': True, 'tree': Directory.find_root(request.user).print()})
    return JsonResponse({'success': False})

@login_required
def delete_item(request):
    if request.method == 'POST':
        item_id = request.POST['item_id']
        type = request.POST['type']
        if item_id and type:
            item = None
            if type == 'file':
                item = File.objects.get(id=item_id)
                if item_id == request.session.get('file_id'): request.session['file_id'] = None
            elif type == 'directory':
                item = Directory.objects.get(id=item_id)
            if item: item.delete()
        return JsonResponse({'success': True, 'tree': Directory.find_root(request.user).print()})
    return JsonResponse({'success': False})

@login_required
def select_file(request):
    if request.method == 'POST':
        file_id = request.POST['file_id']
        if file_id and File.objects.filter(id=file_id).exists():
            file = File.objects.get(id=file_id)
            if not file.owner == request.user or not file.available:
                return JsonResponse({'success': False})
            request.session['file_id'] = file_id
            return JsonResponse({'success': True, 'content': File.objects.get(id=file_id).content})
    return JsonResponse({'success': False})

@login_required
def save_file(request):
    if request.method == 'POST':
        file_id = request.session.get('file_id')
        if file_id != None and File.objects.filter(id=file_id).exists():
            file = File.objects.get(id=file_id)
            if not file.owner == request.user:
                return JsonResponse({'success': False})
            if not file.available:
                return JsonResponse({'success': False})
            file.content = request.POST['content']
            file.save()
            return JsonResponse({'success': True})
        return JsonResponse({'success': False})
    return JsonResponse({'success': False})

@login_required
def compile(request):
    if request.method == 'GET':
        cstd = request.GET['cstd']
        opt = request.GET['opt']
        proc = request.GET['proc']
        proc_opt = request.GET['proc_opt']

        # compile selected file using sdcc -S
        file_id = request.session.get('file_id')
        if not (file_id and File.objects.filter(id=file_id).exists()):
            return JsonResponse({'success': False})
        file = File.objects.get(id=file_id)

        # we use temp.c instead of filename.c because file might not have .c extension
        try:
            with open('temp.c', 'w') as f:
                f.write(file.content)
        except: return JsonResponse({'success': False})

        response = {}

        cmd = ['sdcc', '-S']
        if proc != '': cmd.append(proc)
        if proc_opt != '': cmd.extend(proc_opt.split(' '))
        if cstd != '': cmd.append(cstd)
        if opt != '': cmd.extend(opt.split(' '))
        cmd.append('temp.c')

        # run subprocess save stderr to response
        try: response['stderr'] = subprocess.run(cmd, stderr=subprocess.PIPE).stderr.decode('utf-8')
        except: return JsonResponse({'success': False})

        try:
            with open('temp.asm', 'r') as f:
                response['asm'] = f.read()
        except: return JsonResponse({'success': False})
        response['success'] = True

        temp_files = [
            'temp.c',
            'temp.asm',
            'temp.ihx',
            'temp.lk',
            'temp.lst',
            'temp.map',
            'temp.mem',
            'temp.rel',
            'temp.rst',
            'temp.sym',
        ]
        for temp_file in temp_files:
            subprocess.run(['[ -f ' + temp_file + ' ] && rm ' + temp_file], shell=True)

        response['filename'] = file.name + '.asm'
        return JsonResponse(response)
    return JsonResponse({'success': False})
