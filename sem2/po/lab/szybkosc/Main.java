import java.io.*;
import java.util.*;
public class Main {
  public static void convexHull(ArrayList<Point> p, ArrayList<Point> ow) {
    Collections.sort(p, new Comparator<Point>() { public int compare(Point a, Point b) { return (a.x > b.x || a.x == b.x && a.y > b.y) ? 1 : -1; } });

    for (var pt : p) {
      while (ow.size() > 1 && pt.side(ow.get(ow.size() - 2), ow.get(ow.size() - 1)) == 1)
        ow.remove(ow.size() - 1);
      ow.add(pt);
    }
    
    ow.remove(ow.size() - 1); int k = ow.size();
    Collections.reverse(p);

    for (var pt : p) {
      while (ow.size() > 1 && pt.side(ow.get(ow.size() - 2), ow.get(ow.size() - 1)) == 1)
        ow.remove(ow.size() - 1);
      ow.add(pt);
    }
    
    ow.remove(ow.size() - 1);
  }

  public static void main(String[] args) throws Exception {
    ArrayList<Point> p = new ArrayList<Point>(), ow = new ArrayList<Point>();
    for (long n = getczary(); n != 0; n--)
      p.add(new Point(getczary(), getczary()));
    convexHull(p, ow);
    System.out.printf("%d\n", ow.size());
  }

  public static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
  public static boolean isdigit(int znak) throws Exception {
    return 47 < znak && znak < 58;
  }
  public static long getczary() throws Exception {
    boolean ujemna = false;
    int znak = br.read();
    long wynik = 0;
    while (!isdigit(znak)) {
      if (znak == '-')
        ujemna = true;
      znak = br.read();
    }
    while (isdigit(znak)) {
      wynik *= 10;
      wynik += znak & 15;
      znak = br.read();
    }
    if (ujemna)
      znak *= -1;
    return wynik;
  }
}

class Point {
  public long x, y;
  Point(){}
  Point(long _x, long _y){ x = _x; y = _y; }

  Point      add(Point a) { return new Point(x + a.x, y + a.y); }
  Point subtract(Point a) { return new Point(x - a.x, y - a.y); }
  Point multiply(Point a) { return new Point(x * a.x, y * a.y); }

  long crossProduct(Point a) { return x * a.y - y * a.x; }
  long   dotProduct(Point a) { return x * a.x + y * a.y; }

  int side(Point a, Point b) {
    long orientation = (b.subtract(a)).crossProduct(this.subtract(a));
    return orientation == 0 ? 0 : (orientation > 0 ? 1 : 2);
  }
}
