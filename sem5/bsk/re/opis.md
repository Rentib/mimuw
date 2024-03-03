# BSK - Inżynieria Wsteczna

## Odmaskowanie hasła

Używając *IDA* znalazłem *pc\_putch*, które -- jak sama nazwa wskazuje --
zajmuje się wypisywaniem znaków w komputerze. Tam znalazłem instrukcję, która
sprawia, że wypisywana jest gwiazdka, czyli znak o kodzie ASCII *2Ah*. Aby
sprawdzić, czy rzeczywiście tak jest, podmieniłem kod na inny i rzeczywiście
zmienił się też wypisywany znak. Następnie, używając *x86dbg*, sprawdziłem,
które rejestry zmieniają się w zależności od tego, co wpiszę na komputerze.
Okazało się, że znak jest trzymany w rejestrze *r9*. Zamiast kodu ASCII podałem
więc do instrukcji rejestr *r9w*, i dopełniłem instrukcję NOP-em, aby miała
tyle samo bajtów.

## Jak sprawdzane jest hasło do dysku?

Hasło jest sprawdzane w procedurze *pc\_check\_luks\_password*. Jeżeli ma inną
długość niż 8, to nie zostaje sprawdzone. Następnie sprawdzane jest, czy na
pozycjach zapalonych bitów liczby *0x91* nie występuje cyfra. Jeśli występuje,
to hasło jest uznawane za błędne. Następnie, w przypadku gdy wszystko się
zgadzało, obliczany jest hasz hasła za pomocą funkcji *hash\_password*. Funkcja
ta korzysta z zadanej tablicy 64-bitowych liczb. Otrzymany hasz jest
porównywany z haszem poprawnego hasła. Jeśli są one równe, hasło jest uznawane
za poprawne. Słowne opisywanie algorytmu nie ma zbytnio sensu, więc do
rozwiązania dołączam plik *c* implementujący obie funkcje. Poprawne hasło można
uzyskać brutalnie sprawdzając każdą możliwą kombinację cyfr i liter. Dodatkowo,
przyjąłem, że jeśli na pewnym miejscu może być cyfra, to jest tam cyfra. To
znacząco zmniejszyło liczbę haseł do sprawdzenia: z *36^8* do *26^3 \cdot 10^5*
i na szczęście okazało się to być dobrym założeniem. Jeśli posiadałbym kartę
graficzną wspierającą CUDA, to użyłbym jej, ale jej nie mam, więc odpaliłem kod
na *26* wątkach -- po jednym na każdą możliwą pierwszą literkę. Teraz każdy
wątek miał do sprawdzenia jedynie *26^2 \cdot 10^5* haseł, co zajęło nieco
ponad sekundę. Poprawnym hasłem okazało się być *p455w04d*. Do rozwiązania
załączam też kod implementujący to rozwiązanie -- plik *crack.c*.

## Mame nie teraz, są nowe poszlaki

Można zauważyć, że mama zaczyna nas denerwować tylko jeśli procedura
*retask.check*(1) w procedurze *on\_enter* zwróci *0*. Procedura ta sprawdza,
czy ustawiony jest *c*-ty bit pewnej wartości. Ta wartość jest właśnie stanem
gry, o którym mowa w poleceniu. Okazuje się, że jeśli zagramy w grę tak jak
twórcy oczekiwali, czyli zejdziemy po schodach, porozmawiamy z mamą
i spróbujemy wpisać hasło, to stan jest równy *0x575B* (po wpisaniu poprawnego
hasła). Aby zmienić stan na właśnie taki, zmieniłem część procedury *SDL\_main*
wykonującą funkcje *mark* na taką, która po prostu ustawia wartość stanu gry na
odpowiednią. Standardowo dodajemy instrukcje NOP, aby zachować rozmiar pliku.
Warto zauważyć, że po wpisaniu poprawnego hasła nie jest możliwe korzystanie
z komputera, a co za tym idzie, nie da się zweryfikować poprawności pierwszego
podzadania poprzez uruchomienie gry. Stan można także zmienić na taki, jak po
3-krotnym podaniu niepoprawnego hasła i wtedy można nadal korzystać
z komputera, ale w poleceniu nie jest sprecyzowane, której wersji użyć.

## Nieuczciwa przewaga

Metodą jakiej użyłem jest uniemożliwienie Garemu użycia potek (*ang. potion*).
W procedurze *strategy_endless_healing*, której Gary używa do walki jest pewien
warunek, który przy spełnieniu pozwala mu uleczyć swojego pokemona. My nie mamy
takiej możliwości, więc aby wyrównać szanse zabrałem tę możliwość Garemu
poprzez zmianę warunkowego skoku na skok bezwarunkowy. Warto zauważyć, że jest
też możliwość zmiany jego strategii na *strategy_simple*, ale wówczas Gary
traci możliwość wykonywania specjalnych ruchów takich jak *growl*, co daje nam
znaczącą przewagę, a co za tym idzie; rozwiązanie takie jest niewłaściwe.
