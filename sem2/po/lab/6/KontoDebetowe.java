public class KontoDebetowe extends Konto {
  protected final double dopuszczalnyDebet;
  protected double kara;
  public static final double STOPA = 0.10;

  public KontoDebetowe(String _właściciel, double _saldo, double _dopuszczalnyDebet) {
    super(_właściciel, _saldo);
    dopuszczalnyDebet = _dopuszczalnyDebet;
    kara = 0;
  }

  @Override
  public void wpłata(double kwota) {
    sumaWpłat += kwota;
    sumaSald -= saldo;
    if (kwota <= kara) {
      zysk += kwota;
      kara -= kwota;
    } else {
      kwota -= kara;
      zysk += kara;
      kara = 0;
      saldo += kwota;
    }
    sumaSald += saldo;
  }

  public boolean wypłata(double kwota) {
    if (kwota - saldo > dopuszczalnyDebet)
      return false;
    sumaSald -= saldo;
    sumaWypłat += kwota;
    if (saldo <= 0)
      kara += kwota * STOPA;
    else if (saldo < kara)
      kara += (kara - saldo) * STOPA;
    saldo -= kwota;
    sumaSald += saldo;
    return true;
  }

  @Override
  public String toString() {
    StringBuilder s = new StringBuilder();
    s.append(super.toString()).append(", " + kara);
    return s.toString();
  }
}
