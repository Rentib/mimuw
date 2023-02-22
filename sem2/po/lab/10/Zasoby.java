class MójBłąd extends Exception {
  private String plik;

  MójBłąd(String plik) {
    this.plik = plik;
  }

  @Override
  public String toString() {
    return super.toString() + "(" + plik + ")";
  }
}

class BłądOtwierania extends MójBłąd {
  BłądOtwierania(String plik) {
    super(plik);
  }
}

class BłądCzytania extends Exception {
  BłądCzytania(String plik) {
    super(plik);
  }
}

class Zasób1 implements AutoCloseable {
  private boolean czyMa = false;
  private String nazwa;

  public Zasób1(String nazwa) throws BłądOtwierania {
    if (Math.random() > 0.75)
      throw new BłądOtwierania(nazwa);
    this.nazwa = nazwa;
    System.out.printf("Zasób1(%s) otwarty.\n", nazwa);
  }

  public boolean maLiczbę() {
    return czyMa = Math.random() > 0.5;
  }

  public int dajLiczbę() throws BłądCzytania {
    if (!czyMa || Math.random() > 0.9)
      throw new BłądCzytania(nazwa);
    else
      return (int) (Math.random() * 42);
  }

  @Override
  public void close() {
    System.out.printf("Zasób1(%s) zamknięty!\n", nazwa);
  }
}

class Zasób2 implements AutoCloseable {

  public Zasób2() {
    System.out.println("Zasób2 otwarty.");
  }


  public void zróbCoś() {
  }

  @Override
  public void close() {
    System.out.println("Zasób2 zamknięty!");
  }

}

public class Zasoby {

  public int m(String[] nazwyPlików, int k) throws Exception {
    Zasób2 z2 = new Zasób2();
    if (k == 0)
      return -1;

    for (int i = 0; i < nazwyPlików.length; i++) {
      Zasób1 z = new Zasób1(nazwyPlików[i]);

      while (z.maLiczbę()) {
        int wyn = z.dajLiczbę();
        if (wyn % k == 0) {
          return wyn;
        }
      }

    }
    return 0;
  }

  // try-catch
  public int m1(String[] nazwyPlików, int k) throws BłądOtwierania, BłądCzytania {
    if (k == 0)
      return -1;
    Zasób2 z2 = new Zasób2();
    try {
      for (int i = 0; i < nazwyPlików.length; i++) {
        Zasób1 z = new Zasób1(nazwyPlików[i]);
        try {
          while (z.maLiczbę()) {
            int wyn = z.dajLiczbę();
            if (wyn % k == 0) {
              z.close();
              z2.close();
              return wyn;
            }
          }
        } catch (BłądCzytania e) {
          z.close();
          throw e;
        }
        z.close();
      }
    } catch (BłądOtwierania|BłądCzytania e) {
      z2.close();
      throw e;
    }

    z2.close();

    return 0;
  }

  // try-finally
  public int m2(String[] nazwyPlików, int k) throws BłądOtwierania, BłądCzytania {
    if (k == 0)
      return -1;
    Zasób2 z2 = new Zasób2();

    try {
      for (int i = 0; i < nazwyPlików.length; i++) {
        Zasób1 z = new Zasób1(nazwyPlików[i]);

        try {
          while (z.maLiczbę()) {
            int wyn = z.dajLiczbę();
            if (wyn % k == 0) {
              return wyn;
            }
          }
        } finally {
          z.close();
        }

      }
    } finally {
      z2.close();
    }

    return 0;
  }

  // try z zasobami
  public int m3(String[] nazwyPlików, int k) throws BłądOtwierania, BłądCzytania {
    try (Zasób2 z2 = new Zasób2()) {
      if (k == 0)
        return -1;

      for (int i = 0; i < nazwyPlików.length; i++) {
        try (Zasób1 z = new Zasób1((nazwyPlików[i]))) {
          while (z.maLiczbę()) {
            int wyn = z.dajLiczbę();
            if (wyn % k == 0) {
              return wyn;
            }
          }
        }
      }
    }

    return 0;
  }

  public static void main(String[] args) {
    Zasoby z = new Zasoby();

    System.out.printf("m1\n");
    try {
      z.m1(new String[]{"raz", "dwa", "trzy", "cztery"}, 13);
    } catch (BłądOtwierania|BłądCzytania e) {
      System.out.println(e);
    }

    System.out.printf("m2\n");
    try {
      z.m2(new String[]{"raz", "dwa", "trzy", "cztery"}, 13);
    } catch (BłądOtwierania|BłądCzytania e) {
      System.out.println(e);
    }

    System.out.printf("m3\n");
    try {
      z.m3(new String[]{"raz", "dwa", "trzy", "cztery"}, 13);
    } catch (BłądOtwierania|BłądCzytania e) {
      System.out.println(e);
    }
  }
}
