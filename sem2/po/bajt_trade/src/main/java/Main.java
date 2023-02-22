import java.util.*;
import java.io.*;
import java.nio.file.*;
import com.squareup.moshi.*;

public class Main {
  public static void main(String[] args) throws IOException {
    assert(args.length > 0);
    String path = args[0];
    File wejscie = new File(path);
    String json = new String(Files.readAllBytes(Path.of(wejscie.getPath())));
    Moshi moshi = new Moshi.Builder().build();
    JsonAdapter<Symulacja> jsonAdapter = moshi.adapter(Symulacja.class);
    Symulacja symulacja = jsonAdapter.fromJson(json);
  }
}

class Symulacja {
  private Gielda info;
  private List<Robotnik> robotnicy;
  private List<Spekulant> spekulanci;

  public Symulacja() {
    info       = new Gielda();
    robotnicy  = new ArrayList<Robotnik>();
    spekulanci = new ArrayList<Spekulant>();
  }

  public void sortuj_robotnikow() {
    int cmp = info.mnoznik_komparatora();
    robotnicy.sort(new Comparator<Robotnik>() {
      @Override public int compare(Robotnik a, Robotnik b) {
        return a.diamenty() - b.diamenty() < 0 ? -cmp : cmp;
      }
    });
  }
}

class Gielda {
  private static enum RodzajGieldy {
    @Json(name = "kapitalistyczna") KAPITALISTYCZNA,
    @Json(name =  "socjalistyczna") SOCJALISTYCZNA, 
    @Json(name =    "zrownowazona") ZROWNOWAZONA,
  }
  private static class Ceny {
    public double jedzenie;
    public double ubrania;
    public double narzedzia;
    public double programy;
    public Ceny() {
      jedzenie  = 0;
      ubrania   = 0;
      narzedzia = 0;
      programy  = 0;
    }
  }

  private final int dlugosc;
  private final RodzajGieldy gielda;
  private final int kara_za_brak_ubran;
  private Ceny ceny;

  private int dzien;
  private List<Ceny> historia_cen;

  public Gielda() {
    dlugosc = 0;
    gielda  = RodzajGieldy.KAPITALISTYCZNA;
    kara_za_brak_ubran = 0;
    ceny = null;

    dzien = 1;
    historia_cen = new ArrayList<Ceny>();
  }

  public int dzien() { return dzien; }
  public int mnoznik_komparatora() {
    switch (gielda) {
      case KAPITALISTYCZNA: return -1;
      case SOCJALISTYCZNA:  return +1;
      case ZROWNOWAZONA:    return dzien % 2 == 1 ? 1 : -1;
    }
    return 0;
  }
  public int wybierz_przedmiot_o_maksymalnej_cenie(int liczba_dni) {
    assert(dzien >= liczba_dni);
    double maksymalna_cena = 0;
    int produkt = 0;
    for (int i = historia_cen.size(); liczba_dni > 0; i--, liczba_dni--) {
      if (historia_cen.get(i).jedzenie > maksymalna_cena) {
        maksymalna_cena = historia_cen.get(i).jedzenie;
        produkt = 0;
      }
      if (historia_cen.get(i).ubrania > maksymalna_cena) {
        maksymalna_cena = historia_cen.get(i).jedzenie;
        produkt = 2;
      }
      if (historia_cen.get(i).narzedzia > maksymalna_cena) {
        maksymalna_cena = historia_cen.get(i).narzedzia;
        produkt = 3;
      }
      if (historia_cen.get(i).programy > maksymalna_cena) {
        maksymalna_cena = historia_cen.get(i).programy;
        produkt = 4;
      }
    }
    return produkt;
  }
  public int wybierz_przedmiot_o_maksymalnym_wzroscie(int liczba_dni) {
    assert(dzien >= liczba_dni);
    int a = dzien - 1;
    int b = dzien - liczba_dni;

    double maks_wzrost = historia_cen.get(a).jedzenie - historia_cen.get(b).jedzenie;
    int produkt     = 0;

    if (historia_cen.get(a).ubrania - historia_cen.get(b).ubrania > maks_wzrost) {
      maks_wzrost = historia_cen.get(a).ubrania - historia_cen.get(b).ubrania;
      produkt = 0;
    }
    if (historia_cen.get(a).narzedzia - historia_cen.get(b).narzedzia > maks_wzrost) {
      maks_wzrost = historia_cen.get(a).narzedzia - historia_cen.get(b).narzedzia;
      produkt = 0;
    }
    if (historia_cen.get(a).programy - historia_cen.get(b).programy > maks_wzrost) {
      maks_wzrost = historia_cen.get(a).programy - historia_cen.get(b).programy;
      produkt = 0;
    }

    return produkt;
  }
}

abstract class Agent {
  protected static enum RodzajProduktu {
    JEDZENIE, DIAMENTY, UBRANIA, NARZEDZIA, PROGRAMY;
    public static RodzajProduktu fromInteger(int n) {
      switch(n) {
        case 0: return JEDZENIE;
        case 1: return DIAMENTY;
        case 2: return UBRANIA;
        case 3: return NARZEDZIA;
        case 4: return PROGRAMY;
        default: return null;
      }
    }
  }
  protected static class Produkty {
    public int jedzenie;
    public double diamenty;
    public int ubrania;
    public int narzedzia;
    public int programy;

    public List<Integer> lista_ubran;
    public List<Integer> lista_narzedzi;
    public List<Integer> lista_programow;

    public Produkty() {
      jedzenie  = 0;
      diamenty  = 0;
      ubrania   = 0;
      narzedzia = 0;
      programy  = 0;  

      lista_ubran     = new ArrayList<Integer>();
      lista_narzedzi  = new ArrayList<Integer>();
      lista_programow = new ArrayList<Integer>();
    }

    public int jedzenie()  { return jedzenie;  }
    public double diamenty()  { return diamenty;  }
    public int ubrania()   { return ubrania;   }
    public int narzedzia() { return narzedzia; }
    public int programy()  { return programy;  }
  }
}

class Robotnik extends Agent {
  private static enum RodzajKariery {
    @Json(name = "rolnik")      ROLNIK,
    @Json(name = "gornik")      GORNIK,
    @Json(name = "rzemieslnik") RZEMIESLNIK,
    @Json(name = "inzynier")    INZYNIER,
    @Json(name = "programista") PROGRAMISTA,
  }
  private static class DetaleKupowania {
    public static enum RodzajNabywcy {
      @Json(name = "technofob")      TECHNOFOB,
      @Json(name = "czyscioszek")    CZYSCIOSZEK,
      @Json(name = "zmechanizowany") ZMECHANIZOWANY,
      @Json(name = "gadzeciarz")     GADZECIARZ,
    }
    public final RodzajNabywcy typ;
    public final int liczba_narzedzi;
    public DetaleKupowania() {
      typ             = RodzajNabywcy.TECHNOFOB;
      liczba_narzedzi = 0;
    }
  }
  private static class DetaleProdukcji {
    public static enum RodzajProducenta {
      @Json(name = "krotkowzroczny")  KROTKOWZROCZNY,
      @Json(name = "chciwy")          CHCIWY,
      @Json(name = "sredniak")        SREDNIAK,
      @Json(name = "perspektywiczny") PERSPEKTYWICZNY,
      @Json(name = "losowy")          LOSOWY,
    }
    public final RodzajProducenta typ;
    public final int historia_sredniej_produkcji;
    public final int historia_perspektywy;
    public DetaleProdukcji() {
      typ                         = RodzajProducenta.KROTKOWZROCZNY;
      historia_sredniej_produkcji = 0;
      historia_perspektywy        = 0;
    }
  }
  private static class DetaleUczenia {
    public static enum RodzajUcznia {
      @Json(name = "pracus")     PRACUS,
      @Json(name = "oszczedny")  OSZCZEDNY,
      @Json(name = "student")    STUDENT,
      @Json(name = "okresowy")   OKRESOWY,
      @Json(name = "rozkladowy") ROZKLADOWY,
    }
    public RodzajUcznia typ;
    public double limit_diamentow;
    public int zapas;
    public int okres;
    public int okresowosc_nauki;
    public DetaleUczenia() {
      typ              = RodzajUcznia.PRACUS;
      limit_diamentow  = 0;
      zapas            = 0;
      okres            = 0;
      okresowosc_nauki = 0;
    }
  }
  private static enum RodzajZmiany {
    @Json(name = "konserwatysta")  KONSERWATYSTA,
    @Json(name = "rewolucjonista") REWOLUCJONISTA,
  }
  private final int id;
  private int poziom; // poziom obecnej kariery
  private int[] poziomy_karier;
  private RodzajKariery kariera;
  private final DetaleKupowania kupowanie;
  private final DetaleProdukcji produkcja;
  private final DetaleUczenia uczenie;
  private final RodzajZmiany zmiana;
  private final Produkty produktywnosc; // wektor bazowy produktywnosci
  private Produkty zasoby;              // poziadane zasoby
  public static Random losuj;

  public Robotnik() {
    id            = 0;
    poziom        = 0;
    kariera       = RodzajKariery.ROLNIK;
    kupowanie     = new DetaleKupowania();
    produkcja     = new DetaleProdukcji();
    uczenie       = new DetaleUczenia();
    zmiana        = RodzajZmiany.KONSERWATYSTA;
    produktywnosc = new Produkty();
    zasoby        = new Produkty();

    poziomy_karier = new int[5];
    Arrays.fill(poziomy_karier, 1);
    losuj = new Random();
  }

  public double diamenty() { return zasoby.diamenty; }

  public double premia(RodzajProduktu produkt) {
    double wynik = 0;
    if (kariera.ordinal() == produkt.ordinal()) {
      switch (poziom) {
        case 1: wynik = 0.5; break;
        case 2: wynik = 1.5; break;
        case 3: wynik = 3.0; break;
        default: wynik = 3.0 + (poziom - 3) / 4; break;
      }
    }
    // TODO
    // głód, brak ubrań, narzędzia
    return wynik;
  }

  public boolean czy_sie_uczy(final Gielda info) {
    switch (uczenie.typ) {
      case PRACUS:
        return false;
      case OSZCZEDNY:
        return zasoby.diamenty() >= uczenie.limit_diamentow;
      case STUDENT:
        // TODO
        return true;
      case OKRESOWY:
        return info.dzien() % uczenie.okres == 0;
      case ROZKLADOWY:
        return losuj.nextInt(info.dzien() + 3) == 0;
      default:
        return false;
    }
  }

  public void ucz_sie() {
    RodzajKariery nowa_kariera = kariera;
    switch (zmiana) {
      case KONSERWATYSTA:
        break;
      case REWOLUCJONISTA:
        int n = Math.max(1, id % 17);
        // TODO
        break;
    }

    if (nowa_kariera != kariera) {
      poziomy_karier[kariera.ordinal()] = poziom; // przypisujemy obecjej karierze nasz poziom
      kariera = nowa_kariera;                     // zmieniamy kariere
      poziom = poziomy_karier[kariera.ordinal()]; // zmieniamy nasz poziom na poziom nowej kariery
    } else {
      poziom++;
    }
  }

  public RodzajProduktu wybierz_produkowany_przedmiot(final Gielda info) {
    switch (produkcja.typ) {
      case KROTKOWZROCZNY:
        return RodzajProduktu.fromInteger(
            info.wybierz_przedmiot_o_maksymalnej_cenie(
              Math.min(info.dzien(), 1)));
      case CHCIWY:
        // TODO
        break;
      case SREDNIAK:
        return RodzajProduktu.fromInteger(
            info.wybierz_przedmiot_o_maksymalnej_cenie(
              Math.min(info.dzien(), produkcja.historia_sredniej_produkcji)));
      case PERSPEKTYWICZNY:
        return RodzajProduktu.fromInteger(
            info.wybierz_przedmiot_o_maksymalnym_wzroscie(
              Math.min(info.dzien(), produkcja.historia_perspektywy)));
      case LOSOWY:
        return RodzajProduktu.fromInteger(losuj.nextInt(5));
    }
    return null;
  }
}

class Spekulant extends Agent {
  private static enum RodzajKariery {
    @Json(name = "sredni")     SREDNI,
    @Json(name = "wypukly")    WYPUKLY,
    @Json(name = "regulujacy") REGULUJACY
  }
  private final int id;
  private final RodzajKariery kariera;
  private Produkty zasoby;

  public Spekulant() {
    id      = 0;
    kariera = RodzajKariery.SREDNI;
    zasoby  = new Produkty();
  }
}
