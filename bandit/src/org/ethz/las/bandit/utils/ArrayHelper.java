package org.ethz.las.bandit.utils;

public class ArrayHelper {

  public static double[] stringArrayToDoubleArray(String[] s, int l, int r) {
    double[] ret = new double[r - l + 1];
    for (int i = l; i <= r; ++i)
      ret[i - l] = Double.parseDouble(s[i]);

    return ret;
  }

  public static int[] stringArrayToIntArray(String[] s, int l, int r) {
    int[] ret = new int[r - l + 1];
    for (int i = l; i <= r; ++i)
      ret[i - l] = Integer.parseInt(s[i]);

    return ret;
  }
}
