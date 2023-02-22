import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Chess {
  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    Position pos = new Position();
    System.out.println(pos);
    while (pos.isOk() && pos.doMove()) {
      sc.nextLine();
      System.out.printf("\033[H\033[2]\n%s\n", pos);
    }
  }
}

class Position {
  static boolean WHITE = false, BLACK = true;
  static Random rand = new Random();
  private boolean side;
  private Piece[] board;
  
  public Position() {
    side = WHITE;
    board = new Piece[]{
      new Rook(BLACK), new Knight(BLACK), new Bishop(BLACK), new Queen(BLACK), new King(BLACK), new Bishop(BLACK), new Knight(BLACK), new Rook(BLACK),
      new Pawn(BLACK), new   Pawn(BLACK), new   Pawn(BLACK), new  Pawn(BLACK), new Pawn(BLACK), new   Pawn(BLACK), new   Pawn(BLACK), new Pawn(BLACK),
                 null,              null,              null,             null,            null,              null,              null,            null,
                 null,              null,              null,             null,            null,              null,              null,            null,
                 null,              null,              null,             null,            null,              null,              null,            null,
                 null,              null,              null,             null,            null,              null,              null,            null,
      new Pawn(WHITE), new   Pawn(WHITE), new   Pawn(WHITE), new  Pawn(WHITE), new Pawn(WHITE), new   Pawn(WHITE), new   Pawn(WHITE), new Pawn(WHITE),
      new Rook(WHITE), new Knight(WHITE), new Bishop(WHITE), new Queen(WHITE), new King(WHITE), new Bishop(WHITE), new Knight(WHITE), new Rook(WHITE),
    };
  }

  boolean side() { return side; }
  boolean isPiece(int sq) { return board[sq] != null; }
  boolean isPiece(Square sq) { return isPiece(sq.toInt()); }
  boolean isEnemy(Square sq) { return isPiece(sq.toInt()) && board[sq.toInt()].getColor() != side; }
  boolean isAlly(Square sq) { return isPiece(sq.toInt()) && board[sq.toInt()].getColor() == side; }

  public boolean isOk() {
    int kcnt = 0;
    for (int sq = 0; sq < 64; sq++) {
      if (!isPiece(sq)) continue;
      if (board[sq].getClass() == King.class)
        kcnt++;
    }
    return kcnt == 2;
  }

  @Override
  public String toString() {
    String sep = "  +---+---+---+---+---+---+---+---+\n";
    String xd = sep;
    for (int rank = 8; rank >= 1; rank--) {
      xd += rank;
      for (int file = 1; file <= 8; file++) {
        int sq = new Square(file, rank).toInt();
        xd += " | ";
        if (board[sq] != null)
          xd += board[sq].toString();
        else
          xd += " ";
      }
      xd += " |\n" + sep;
    }
    xd += "    a   b   c   d   e   f   g   h\n\n";
    xd += "    Side to move: " + (side ? "BLACK" : "WHITE");
    xd += "\n";
    return xd;
  }

  public boolean doMove() {
    ArrayList<Integer> xd = new ArrayList<Integer>();
    for (int i = 0; i < 64; i++) xd.add(i);
    while (xd.size() > 0) {
      int idx = rand.nextInt(xd.size());
      int from = xd.get(idx);
      xd.remove(idx);
      Piece p = board[from];
      if (!isPiece(from) || p.getColor() != side) continue;

      ArrayList<Integer> moves = board[from].getMoves(from, this);
      if (moves.size() == 0) continue;

      int to = moves.get(rand.nextInt(moves.size()));
      board[to] = board[from];
      board[from] = null;

      // handling promotions
      if (board[to].getClass() == Pawn.class && to / 8 == 0)
        board[to] = new Queen(side);
      if (board[to].getClass() == Pawn.class && to / 8 == 7)
        board[to] = new Queen(side);

      side = !side;

      return true;
    }
    return false;
  }
}

class Square {
  public int file;
  public int rank;
  public Square(int sq) { rank = 8 - sq / 8; file = sq % 8 + 1; }
  public Square(int x, int y) { file = x; rank = y; }
  public Square add(Square sq) { return new Square(file + sq.file, rank + sq.rank); }
  public int toInt() { return 63 - rank * 8 + file; }
  public boolean isValid() { return 1 <= file && file <= 8 
                                 && 1 <= rank && rank <= 8; }
  public String toString() {
    String f[] = new String[]{ "0", "a", "b", "c", "d", "e", "f", "g", "h" };
    String r[] = new String[]{ "0", "1", "2", "3", "4", "5", "6", "7", "8" };
    return f[file] + r[rank];
  }
}

abstract class Piece {
  protected boolean color;
  public boolean getColor() { return color; }
  abstract public ArrayList<Integer> getMoves(int sq, Position pos);
}

class Pawn extends Piece {
  public Pawn(boolean _color) { color = _color; }
  @Override public String toString() { return color ? "p" : "P"; }

  public ArrayList<Integer> getMoves(int sq, Position pos) {
    ArrayList v = new ArrayList<Integer>();
    Square up = color ? new Square(0, -1) : new Square(0, 1);
    Square upLeft  = up.add(new Square(-1, 0));
    Square upRight = up.add(new Square(1, 0));

    // push
    Square singlePush = up.add(new Square(sq));
    if (singlePush.isValid() && !pos.isPiece(singlePush)) {
      v.add(singlePush.toInt());
      // double push
      Square doublePush = singlePush.add(up);
      if (doublePush.isValid() && !pos.isPiece(doublePush) 
      &&  doublePush.rank == (pos.side() ? 5 : 4))
        v.add(doublePush.toInt());
    }
    
    // left capture
    upLeft = upLeft.add(new Square(sq));
    if (upLeft.isValid() && pos.isEnemy(upLeft))
      v.add(upLeft.toInt());

    // right capture
    upRight = upRight.add(new Square(sq));
    if (upRight.isValid() && pos.isEnemy(upRight))
      v.add(upRight.toInt());

    return v;
  }
}

abstract class Leaper extends Piece {
  protected Square[] directions;

  public ArrayList<Integer> getMoves(int sq, Position pos) {
    ArrayList v = new ArrayList<Integer>();
    Square from = new Square(sq);
    for (Square move : directions) {
      Square to = from.add(move);
      if (to.isValid() && !pos.isAlly(to)) {
        v.add(to.toInt());
        to = to.add(move);
      }
    }
    return v;
  }
}

class King extends Leaper {
  public King(boolean _color) {
    color = _color;
    directions = new Square[8];
    for (int idx = 0, i = -1; i <= 1; i++) { for (int j = -1; j <= 1; j++) { if (i == 0 && j == 0) continue; directions[idx++] = new Square(i, j); } }
  }
  @Override public String toString() { return color ? "k" : "K"; }
}

class Knight extends Leaper {
  public Knight(boolean _color) {
    color = _color;
    directions = new Square[8];
    directions[0] = new Square(-2, -1);
    directions[1] = new Square( 2, -1);
    directions[2] = new Square(-2,  1);
    directions[3] = new Square( 2,  1);

    directions[4] = new Square(-1, -2);
    directions[5] = new Square( 1, -2);
    directions[6] = new Square(-1,  2);
    directions[7] = new Square( 1,  2);
  }
  @Override public String toString() { return color ? "n" : "N"; }
}

abstract class Slider extends Piece {
  protected Square[] directions;
  public ArrayList<Integer> getMoves(int sq, Position pos) {
    ArrayList v = new ArrayList<Integer>();
    Square from = new Square(sq);

    for (Square move : directions) {
      Square to = from.add(move);
      while (to.isValid() && !pos.isPiece(to)) {
        v.add(to.toInt());
        to = to.add(move);
      }
      if (to.isValid() && pos.isEnemy(to))
        v.add(to.toInt());
    }

    return v;
  }
}

class Bishop extends Slider {
  public Bishop(boolean _color) {
    color = _color;
    directions = new Square[4];
    directions[0] = new Square(-1, -1);
    directions[1] = new Square( 1, -1);
    directions[2] = new Square(-1,  1);
    directions[3] = new Square( 1,  1);
  }
  @Override public String toString() { return color ? "b" : "B"; }
}

class Rook extends Slider {
  public Rook(boolean _color) {
    color = _color;
    directions = new Square[4];
    directions[0] = new Square(-1,  0);
    directions[1] = new Square( 1,  0);
    directions[2] = new Square( 0, -1);
    directions[3] = new Square( 0,  1);
  }
  @Override public String toString() { return color ? "r" : "R"; }
}

class Queen extends Slider {
  public Queen(boolean _color) {
    color = _color;
    directions = new Square[8];
    for (int idx = 0, i = -1; i <= 1; i++) { for (int j = -1; j <= 1; j++) { if (i == 0 && j == 0) continue; directions[idx++] = new Square(i, j); } }
  }
  @Override public String toString() { return color ? "q" : "Q"; }
}
