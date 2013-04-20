package org.ethz.las;

import java.util.*;

public class HashFamily {

  public static SimpleHashFunction[] generateSimpleHashFamily(int maxCoef, int mod, int nHashFunctions, int seed) {
    Random R = new Random(seed);
    SimpleHashFunction[] hashFunctions = new SimpleHashFunction[nHashFunctions];
    for (int i = 0; i < nHashFunctions; ++i) {
      int a = R.nextInt(maxCoef);
      int b = R.nextInt(maxCoef) + 1;
      hashFunctions[i] = new SimpleHashFunction(a, b, mod);
    }
    return hashFunctions;
  }

  public static VectorHashFunction[] generateVectorHashFamily(int maxCoef, int length, int nHashFunctions, int seed) {
    Random R = new Random(seed);
    VectorHashFunction[] hashFunctions = new VectorHashFunction[nHashFunctions];

    for (int i = 0; i < nHashFunctions; ++i) {
      int [] a = new int[length];
      int b = R.nextInt(maxCoef) + 1;
      for (int j = 0; j < length; ++j)
        a[j]= R.nextInt(maxCoef) + 1;

      hashFunctions[i] = new VectorHashFunction(a, b);
    }
    return hashFunctions;
  }
}
