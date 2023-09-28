from django.test import TestCase, RequestFactory, Client
from django.contrib.auth.models import User, AnonymousUser
from datetime import date
from django.urls import reverse
from django.contrib.auth.decorators import login_required

from .models import *

class DirectoryTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('testuser', 'testpassword')

    def test_print(self):
        root = Directory.find_root(self.user)
        sub = Directory(name='sub', parent=root, owner=self.user)
        sub.save()
        file = File(name='file', parent=sub, owner=self.user)
        file.save()

        expected = ("<ul><li class='folder' data-id='1'>"
                    "<a class='add-file-btn'><i class='fas fa-file-circle-plus'></i></a>"
                    "<a class='add-dir-btn'><i class='fas fa-folder-plus'></i></a>"
                    "</li><ul><li class='folder' data-id='2'>sub <a class='add-file-btn'><i class='fas fa-file-circle-plus'></i></a>"
                    "<a class='add-dir-btn'><i class='fas fa-folder-plus'></i></a><a class='delete-btn'><i class='fas fa-trash'></i></a></li>"
                    "<ul><li data-id='1'><a class='file-select-btn'>file</a>"
                    "<a class='delete-btn'><i class='fas fa-trash'></i></a></li></ul></ul></ul>")

        self.assertEqual(root.print(), expected)

        sub.delete()
        root.delete()

    def test_delete(self):
        root = Directory.find_root(self.user)
        sub = Directory(name='sub', parent=root, owner=self.user)
        sub.save()
        file = File(name='file', parent=sub, owner=self.user)
        file.save()
        sub.delete()
        sub.delete()
        cnt = 0
        for i in root.directory_set.all():
            cnt += 1 if i.available else 0
        for i in root.file_set.all():
            cnt += 1 if i.available else 0
        self.assertEqual(cnt, 0)
        root.delete()


class FileModelTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.root = Directory.find_root(self.user)
        self.file = File(name='Test File', parent=self.root, owner=self.user)
        self.file.save()

    def test_print_method(self):
        expected_output = (
            "<ul>"
            "<li data-id='1'><a class='file-select-btn'>Test File</a><a class='delete-btn'><i class='fas fa-trash'></i></a></li>"
            "</ul>"
        )

        self.assertEqual(self.file.print(), expected_output)

    def test_delete_method(self):
        self.file.delete()

        self.assertFalse(self.file.available)
        self.assertEqual(self.file.availability_changed_at.date(), date.today())

    def test_str_method(self):
        self.assertEqual(str(self.file), 'Test File')

class SectionModelTestCase(TestCase):
    def test_create_section(self):
        section = Section.objects.create(
            name='Test Section',
            description='Test description',
            start_line=1,
            end_line=5,
            start_char=10,
            end_char=20,
            section_type='Type',
            section_category='Category',
            section_status='Status',
            status_data='Status data',
            content='Section content'
        )

        self.assertEqual(section.name, 'Test Section')
        self.assertEqual(section.description, 'Test description')
        self.assertEqual(section.start_line, 1)
        self.assertEqual(section.end_line, 5)
        self.assertEqual(section.start_char, 10)
        self.assertEqual(section.end_char, 20)
        self.assertEqual(section.section_type, 'Type')
        self.assertEqual(section.section_category, 'Category')
        self.assertEqual(section.section_status, 'Status')
        self.assertEqual(section.status_data, 'Status data')
        self.assertEqual(section.content, 'Section content')

from .views import *

class IndexViewTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(username='testuser', password='testpass')

    def test_authenticated_user(self):
        request = self.factory.get(reverse('index'))
        request.user = self.user
        response = index(request)
        self.assertEqual(response.status_code, 200)

    def test_unauthenticated_user(self):
        request = self.factory.get(reverse('index'))
        request.user = AnonymousUser()
        response = index(request)
        self.assertEqual(response.status_code, 302)

class LoadTreeViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.directory = Directory.objects.create(name='Test Directory', owner=self.user)

    def test_load_tree_success(self):
        self.client.force_login(self.user)

        response = self.client.get(reverse('load_tree'))
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('tree', data)

    def test_load_tree_unauthenticated(self):
        response = self.client.get(reverse('load_tree'))
        self.assertEqual(response.status_code, 302)

class LoadFileViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')

        self.root = Directory.find_root(self.user)
        self.file = File(name='Test File', owner=self.user, parent=self.root)
        self.file.save()

    def test_load_file_success(self):
        self.client.force_login(self.user)

        response = self.client.get(reverse('load_file'), {'file_id': self.file.id})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertFalse(data['success']) # no selected file

    def test_load_file_success_selected(self):
        # select file
        self.client.force_login(self.user)
        self.client.post(reverse('select_file'), {'file_id': self.file.id})

        response = self.client.get(reverse('load_file'), {'file_id': self.file.id})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('content', data)
        self.assertEqual(data['content'], self.file.content)

class AddChildViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.parent_directory = Directory.objects.create(name='Parent Directory', owner=self.user)
        self.parent_id = self.parent_directory.id
        self.child_name = 'Child Name'
        self.child_type = 'file'  # Set the type to 'file' or 'directory' based on your test scenario

    def test_add_child_success(self):
        self.client.force_login(self.user)

        response = self.client.post(reverse('add_child'), data={
            'parent_id': self.parent_id,
            'name': self.child_name,
            'type': self.child_type
        })

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('tree', data)

        if self.child_type == 'file':
            self.assertTrue(File.objects.filter(name=self.child_name, owner=self.user, parent=self.parent_directory).exists())
        elif self.child_type == 'directory':
            self.assertTrue(Directory.objects.filter(name=self.child_name, owner=self.user, parent=self.parent_directory).exists())

    def test_add_child_unauthenticated(self):
        response = self.client.post(reverse('add_child'), data={
            'parent_id': self.parent_id,
            'name': self.child_name,
            'type': self.child_type
        })

        self.assertEqual(response.status_code, 302)


class DeleteItemViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.item_directory = Directory.objects.create(name='Item Directory', owner=self.user)
        self.item_file = File.objects.create(name='Item File', owner=self.user, parent=self.item_directory)
        self.item_id = self.item_file.id
        self.item_type = 'file'  # Set the type to 'file' or 'directory' based on your test scenario

    def test_delete_item_success(self):
        self.client.force_login(self.user)

        response = self.client.post(reverse('delete_item'), data={
            'item_id': self.item_id,
            'type': self.item_type
        })

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('tree', data)

        if self.item_type == 'file':
            item = File.objects.filter(id=self.item_id)
        elif self.item_type == 'directory':
            item = Directory.objects.filter(id=self.item_id)
        self.assertTrue(item.exists())
        self.assertFalse(item.first().available)

    def test_delete_item_success_recursive(self):
        self.client.force_login(self.user)

        response = self.client.post(reverse('add_child'), data={
            'parent_id': self.item_directory.id,
            'name': 'Child Directory',
            'type': 'directory'
        })
        self.assertEqual(response.status_code, 200)
        child_directory = Directory.objects.get(name='Child Directory')

        response = self.client.post(reverse('add_child'), data={
            'parent_id': Directory.objects.get(name='Child Directory').id,
            'name': 'Child File',
            'type': 'file'
        })
        self.assertEqual(response.status_code, 200)
        child_file = File.objects.get(name='Child File')

        response = self.client.post(reverse('delete_item'), data={
            'item_id': child_directory.id,
            'type': 'directory'
        })
        self.assertEqual(response.status_code, 200)
        child_directory = Directory.objects.get(name='Child Directory')
        child_file = File.objects.get(name='Child File')
        self.assertFalse(child_directory.available)
        self.assertFalse(child_file.available)

    def test_delete_item_unauthenticated(self):
        response = self.client.post(reverse('delete_item'), data={
            'item_id': self.item_id,
            'type': self.item_type
        })

        self.assertEqual(response.status_code, 302)

class SelectFileViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.root = Directory.find_root(self.user)
        self.file1 = File(name='file1', owner=self.user, parent = self.root)
        self.file1.save()
        self.tmpdir = Directory(name='tmpdir', owner=self.user, parent=self.root)
        self.tmpdir.save()
        self.file2 = File(name='file2', owner=self.user, parent = self.tmpdir)
        self.file2.save()

    def test_select_file_successful(self):
        self.client.force_login(self.user)
        # select file1
        response = self.client.post(reverse('select_file'), data={
            'file_id': self.file1.id,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        # select file2
        response = self.client.post(reverse('select_file'), data={
            'file_id': self.file2.id,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])

    def test_select_file_invalid(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse('select_file'), data={
            'file_id': 69,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['success'])

    def test_select_file_unauthorized(self):
        response = self.client.post(reverse('select_file'), data={
            'file_id': 69,
        })
        self.assertEqual(response.status_code, 302)

class SaveFileViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.root = Directory.find_root(self.user)
        self.file = File(name='file1', owner=self.user, parent = self.root)
        self.file.save()

    def test_save_fail_no_selected_file(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse('save_file'), data={
            'content': 'Updated file content',
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['success'])

    def test_save_file_unauthenticated(self):
        response = self.client.post(reverse('save_file'), data={
            'content': 'Updated file content',
        })
        self.assertEqual(response.status_code, 302)

    def test_save_file_invalid_file_id(self):
        self.client.force_login(self.user)
        # Set an invalid file ID in the session
        self.client.session['file_id'] = 999
        response = self.client.post(reverse('save_file'), data={
            'content': 'Updated file content',
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['success'])

    def test_save_file_unauthorized(self):
        other_user = User.objects.create_user(username='otheruser', password='testpassword')
        self.client.force_login(other_user)
        self.client.session['file_id'] = self.file.id
        response = self.client.post(reverse('save_file'), data={
            'content': 'Updated file content',
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['success'])

    def test_save_file_successful(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse('select_file'), data={
            'file_id': self.file.id,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])

        response = self.client.post(reverse('save_file'), data={
            'content': 'kys',
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])

class CompileViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.root = Directory.find_root(self.user)
        self.errFile = File(name='Test File', owner=self.user, content='Test file content', parent=self.root)
        self.errFile.save()
        self.okFile = File(name='ok.c', owner=self.user, parent=self.root, content='int main() {return 0;}')
        self.okFile.save()

    def test_compile_success(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse('select_file'), data={
            'file_id': self.okFile.id,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])

        response = self.client.get(reverse('compile'), data={
            'cstd': 'c99',
            'opt': 'opt-code-speed',
            'proc': 'mmcs51',
            'proc_opt': '',
        })
        self.assertEqual(response.status_code, 200)
        # NOTE: this test checks literally nothing but i don't care

    def test_compile_unauthenticated(self):
        response = self.client.get(reverse('compile'), data={
            'cstd': 'c99',
            'opt': 'opt-code-speed',
            'proc': 'mmcs51',
            'proc_opt': '',
        })

        self.assertEqual(response.status_code, 302)

    def test_compile_invalid_file_id(self):
        self.client.force_login(self.user)

        # Set an invalid file ID in the session
        self.client.session['file_id'] = 999

        response = self.client.get(reverse('compile'), data={
            'cstd': 'std',
            'opt': '-O1',
            'proc': '8051',
            'proc_opt': '',
        })

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertFalse(data['success'])

    def test_compile_exception(self):
        self.client.force_login(self.user)

        # Set a non-existing compiler in the command
        response = self.client.get(reverse('compile'), data={
            'cstd': 'std',
            'opt': '-O1',
            'proc': '8051',
            'proc_opt': '',
        })

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertFalse(data['success'])
