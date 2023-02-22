import java.util.Arrays;
class Dequeue {
  private String[] dequeue;
  private int size;
  private int front, back;
  private int elements;

  public Dequeue() {
    this.dequeue = new String[1];
    this.size = 1;
    this.front = 0;
    this.back = 0;
    this.elements = 0;
  }

  public boolean empty() {
    return this.elements == 0;
  }

  public int size() {
    return this.elements;
  }

  public String front() {
    assert this.elements > 0 : "queue is empty";
    return this.dequeue[this.front];
  }

  public String back() {
    assert this.elements > 0 : "queue is empty";
    return this.dequeue[this.back];
  }

  public void pop_front() {
    assert this.elements > 0 : "queue is empty";
    this.dequeue[this.front] = null;
    this.front = (this.front + 1) & (this.size - 1);
    this.elements--;
  }

  public void pop_back() {
    assert this.elements > 0 : "queue is empty";
    this.dequeue[this.back] = null;
    this.back = (this.back - 1 + this.size) & (this.size - 1);
    this.elements--;
  }

  private void reorganize() {
    int newSize = this.size << 1;
    String[] newDequeue = new String[newSize];
    for (int i = 0, j = this.front; i < this.size; i++, j = (j + 1) & (this.size - 1))
      newDequeue[i] = this.dequeue[j];
    this.dequeue = newDequeue;
    this.front = 0;
    this.back = this.size;
    this.size = newSize;
  }

  public void push_front(String x) {
    if (this.size == this.elements)
      this.reorganize();
    this.front = (this.front - 1 + this.size) & (this.size - 1);
    this.dequeue[this.front] = x;
    this.elements++;
  }

  public void push_back(String x) {
    if (this.size == this.elements)
      this.reorganize();
    this.dequeue[this.back] = x;
    this.back = (this.back + 1) & (this.size - 1);
    this.elements++;
  }

  public void print() {
    System.out.printf("[");
    for (int i = 0, j = this.front; i < this.elements - 1; i++, j = (j + 1) & (this.size - 1))
      System.out.printf("%s, ", this.dequeue[j]);
    System.out.printf("%s]\n", this.dequeue[(this.front + this.elements - 1) & (this.size - 1)]);
  }

  public void printDebug() {
    this.print();
    System.out.println(Arrays.toString(this.dequeue));
  }

  public static void main(String[] args) {
    Dequeue q = new Dequeue();
    q.push_back("A");
    q.printDebug();
    q.push_back("B");
    q.printDebug();
    q.push_back("C");
    q.printDebug();
    q.push_front("D");
    q.printDebug();
    q.push_front("E");
    q.printDebug();
    q.push_back("F");
    q.printDebug();
    q.pop_back();
    q.printDebug();
    q.pop_front();
    q.pop_front();
    q.printDebug();
  }
}
