public abstract class Wyrażenie {
  abstract public double wartość_w_punkcie(double x);
  public double całka(double a, double b) {
    double res = 0;
    for (double x = a; x <= b; x++) res += wartość_w_punkcie(x);
    return res;
  }

  protected final static int PREC_ADDYTYWNA = 10;
  protected final static int PREC_MULTIPLIKATYWNA = 20;
  protected final static int PREC_UNARNA = 30;

  // abstract public Wyrażenie pochodna();
  // abstract public Wyrażenie uproszczenie();
  @Override
  public String toString() { return pretty(new StringBuilder(), 0).toString(); }
  abstract protected StringBuilder pretty(StringBuilder sb, int prec);

  public static void main(String[] args) {
    Wyrażenie a = new Dodaj(new Stała(3), new Zmienna());
    Wyrażenie b = new Odejmij(new Zmienna(), new Stała(3));
    Wyrażenie ab = new Pomnóż(a, b);
    System.out.printf("%s\n", ab.toString());
  }
}

class Zmienna extends Wyrażenie {
  public Zmienna(){};
  public double wartość_w_punkcie(double x) { return x; }
  protected StringBuilder pretty(StringBuilder sb, int prec) { return sb.append("x"); }
}

class Stała extends Wyrażenie {
  protected double c;
  public Stała(double c) { this.c = c; }
  public double wartość_w_punkcie(double x) { return c; }
  protected StringBuilder pretty(StringBuilder sb, int prec) { return sb.append(c); }
}

abstract class WyrażenieDwuArgumentowe extends Wyrażenie {
  protected Wyrażenie lewe;
  protected Wyrażenie prawe;

  protected int precedencja;
  protected String operator;

  public WyrażenieDwuArgumentowe(Wyrażenie lewe, Wyrażenie prawe) { this.lewe = lewe; this.prawe = prawe; }

  abstract protected double oblicz(double l, double p);
  public double wartość_w_punkcie(double x) { return this.oblicz(lewe.wartość_w_punkcie(x), prawe.wartość_w_punkcie(x)); }
  protected StringBuilder pretty(StringBuilder sb, int prec) {
    boolean nawias = this.precedencja < prec;
    if (nawias) sb.append("(");
    lewe.pretty(sb, this.precedencja).append(" " + this.operator + " ");
    prawe.pretty(sb, this.precedencja + 1);
    if (nawias) sb.append(")");
    return sb;
  }
}

class Dodaj extends WyrażenieDwuArgumentowe {
  public Dodaj(Wyrażenie lewe, Wyrażenie prawe) { super(lewe, prawe); precedencja = PREC_ADDYTYWNA; operator = "+"; }
  protected double oblicz(double l, double p) { return l + p; }
}

class Odejmij extends WyrażenieDwuArgumentowe {
  public Odejmij(Wyrażenie lewe, Wyrażenie prawe) { super(lewe, prawe); precedencja = PREC_ADDYTYWNA; operator = "-"; }
  protected double oblicz(double l, double p) { return l + p; }
}
class Pomnóż extends WyrażenieDwuArgumentowe {
  public Pomnóż(Wyrażenie lewe, Wyrażenie prawe) { super(lewe, prawe); precedencja = PREC_MULTIPLIKATYWNA; operator = "*"; }
  protected double oblicz(double l, double p) { return l + p; }
}
class Podziel extends WyrażenieDwuArgumentowe {
  public Podziel(Wyrażenie lewe, Wyrażenie prawe) { super(lewe, prawe); precedencja = PREC_MULTIPLIKATYWNA; operator = "/"; }
  protected double oblicz(double l, double p) { return l + p; }
}
