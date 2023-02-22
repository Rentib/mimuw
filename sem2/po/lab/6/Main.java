public class Main {
  public static Konto[] arr = new Konto[]{
    new Konto("Błaszczykowski", 10),
    new KontoDebetowe("Lewandowski", 30, 20),
    new Konto("Zieliński", 20),
  };
  static void show() {
    System.out.printf("%s ", Konto.raport());
    System.out.printf("[");
    for (var k : arr) 
      System.out.printf("%s, ", k.toString());
    System.out.printf("]\n");
  }
  public static void main(String[] args) {
    show();
    arr[1].wypłata(20);
    arr[1].wypłata(20);
    show();
    arr[1].wypłata(20);
    show();
    arr[1].wypłata(10);
    show();
    arr[1].wpłata(0.50);
    show();
    arr[0].wypłata(10);
    arr[2].wypłata(20);
    show();
  }
}
