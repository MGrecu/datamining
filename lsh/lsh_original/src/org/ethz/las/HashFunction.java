package org.ethz.las;


/**
 * Used to hash scalars when generating signature matrix.
 */
class SimpleHashFunction {

  private int a, b, m;

  public SimpleHashFunction(int a, int b, int m) {
    this.a = a;
    this.b = b;
    this.m = m;
  }

  public long hash(long x) {
    return (a * x + b) % m;
  }
}

/**
 * Used to hash vectors in the LSH step.
 */
class VectorHashFunction {
  private int[] a;
  private int b;

  public VectorHashFunction(int[] a, int b) {
    this.a = a;
    this.b = b;
  }

  public long hashVector(long[] x) {
    long sum = 0;
    // Can overflow for extra large numbers, careful with 'a' and 'b'
    for (int i = 0; i < x.length; ++i)
      sum += a[i] * x[i];

    return sum + b;
  }

  public long hash(long x) {
    return (a[0] * x) + b;
  }
}
