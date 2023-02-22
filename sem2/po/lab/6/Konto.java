public class Konto {
  protected String właściciel;
  protected double saldo;

  public Konto(String _właściciel, double _saldo) {
    assert _saldo >= 0;
    właściciel = _właściciel;
    saldo = _saldo;

    sumaSald += _saldo;
    sumaWpłat += _saldo;
  }

  public void wpłata(double kwota) {
    saldo += kwota;
    sumaWpłat += kwota;
    sumaSald += kwota;
  }

  public boolean wypłata(double kwota) {
    if (saldo < kwota)
      return false;
    saldo -= kwota;
    sumaWypłat += kwota;
    sumaSald -= kwota;
    return true;
  }

  @Override
  public String toString() {
    StringBuilder s = new StringBuilder();
    s.append(właściciel + " : " + saldo);
    return s.toString();
  }

  protected static double sumaSald = 0;
  protected static double sumaWpłat = 0;
  protected static double sumaWypłat = 0;
  protected static double zysk = 0;

  public static String raport() {
    return "(" + sumaSald + ", " + (sumaWpłat - sumaWypłat - zysk) + ")";
  }
}
