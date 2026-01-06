# Graham 扫描

> 原文：[`www.algorithm-archive.org/contents/graham_scan/graham_scan.html`](https://www.algorithm-archive.org/contents/graham_scan/graham_scan.html)

大约在 Jarvis March([../jarvis_march/jarvis_march.html])的同时，R. L. Graham 也在开发一个算法来找到一组随机点的凸包 [[1]](#cite-1)。与 Jarvis March 不同，它是一个 O(n)操作，Graham Scan 是 O(n log n)，其中 n 是点的数量，n 是凸包的大小。这意味着 Graham Scan 的复杂度不是输出敏感的；此外，在某些情况下，Jarvis March 可能更优，这取决于凸包的大小和要包裹的点数。

与 Jarvis March 从最左端点开始不同，Graham scan 从底部开始。然后我们根据底部点、原点和每个其他点之间的角度对点的分布进行排序。排序后，我们逐点检查，寻找凸包上的点，并丢弃任何其他点。我们通过寻找逆时针旋转来完成这项工作。如果三个点之间的角度向内转，则形状显然不是凸的，因此我们可以丢弃该结果。我们可以通过三角函数或使用叉积来找到旋转是否为逆时针，如下所示：

```
function ccw(a::Point, b::Point, c::Point)
    return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x))
end 
```

```
ccw :: Point -> Point -> Point -> Double
ccw (xa, ya) (xb, yb) (xc, yc) = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa) 
```

```
double ccw(struct point a, struct point b, struct point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
} 
```

```
function ccw(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
} 
```

```
def counter_clockwise(p1, p2, p3):
    """Is the turn counter-clockwise?"""
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) >= (p2[1] - p1[1]) * (p3[0] - p1[0]) 
```

```
func counterClockwise(p1, p2, p3 point) bool {
    return (p3.y-p1.y)*(p2.x-p1.x) >= (p2.y-p1.y)*(p3.x-p1.x)
} 
```

```
static double ccw(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
} 
```

```
(defun ccw (p1 p2 p3)
  "Determines if a turn between three points is counterclockwise"
  (-
    (*
      (- (point-y p2) (point-y p1))
      (- (point-x p3) (point-x p1)))
    (*
      (- (point-y p3) (point-y p1))
      (- (point-x p2) (point-x p1))))) 
```

```
double ccw(const point& a, const point& b, const point& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
} 
```

```
data point(x=0, y=0):
    def angle(self, other):
        """Computes the angle between the two points"""
        match point(x1, y1) in other:
            return atan2(y1 - self.y, x1 - self.x) 
```

```
fn counter_clockwise(a: &Point, b: &Point, c: &Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
} 
```

如果这个函数的输出是 0，则点共线。如果输出为正，则点形成一个逆时针的“左”转。如果输出为负，则点形成一个顺时针的“右”转。我们基本上不希望有顺时针旋转，因为这意味着我们处于内角。

为了节省内存和昂贵的`append()`操作，我们最终寻找应该在凸包上的点，并将它们与数组中的第一个元素交换。如果有 n 个元素在凸包上，那么我们输出随机分布的点中的前 n 个元素将是凸包。最后，代码应该看起来像这样：

```
function graham_scan!(points::Vector{Point})
    N = length(points)

    # Place the lowest point at the start of the array
    sort!(points, by = item -> item.y)

    # Sort all other points according to angle with that point
    other_points = sort(points[2:end], by = item -> atan(item.y - points[1].y,
                                                         item.x - points[1].x))

    # Place points sorted by angle back into points vector
    for i in 1:length(other_points)
        points[i+1] = other_points[i]
    end

    # M will be the point on the hull
    M = 2
    for i = 1:N
        while (ccw(points[M-1], points[M], points[i]) <= 0)
            if (M > 2)
                M -= 1
            # All points are collinear
            elseif (i == N)
                break
            else
                i += 1
            end
        end

        # ccw point found, updating hull and swapping points
        M += 1
        points[i], points[M] = points[M], points[i]
    end

    return points[1:M]
end 
```

```
grahamScan :: [Point] -> [Point]
grahamScan [] = []
grahamScan pts = wrap sortedPts [p0]
  where p0@(x, y)= minimumBy (compare `on` snd) pts
        sortedPts = sortOn (\(px, py) -> atan2 (py-y) (px-x) ) $ filter (/=p0) pts
        wrap [] ps = ps
        wrap (s:ss) [p] = wrap ss [s, p]
        wrap (s:ss) (p1:p2:ps)
          | ccw s p1 p2 > 0 = wrap (s:ss) (p2:ps)
          | otherwise       = wrap ss (s:p1:p2:ps) 
```

```
size_t graham_scan(struct point *points, size_t size) {
    qsort(points, size, sizeof(struct point), cmp_points);
    polar_angles_sort(points, points[0], size);

    struct point tmp_points[size + 1];
    memcpy(tmp_points + 1, points, size * sizeof(struct point));
    tmp_points[0] = tmp_points[size];

    size_t m = 1;
    for (size_t i = 2; i <= size; ++i) {
        while (ccw(tmp_points[m - 1], tmp_points[m], tmp_points[i]) <= 0) {
            if (m > 1) {
                m--;
                continue;
            } else if (i == size) {
                break;
            } else {
                i++;
            }
        }

        m++;
        struct point tmp = tmp_points[i];
        tmp_points[i] = tmp_points[m];
        tmp_points[m] = tmp;
    }

    memcpy(points, tmp_points + 1, size * sizeof(struct point));

    return m;
} 
```

```
function grahamScan(points) {
  // First, sort the points so the one with the lowest y-coordinate comes first (the pivot)
  points = [...points].sort((a, b) => (a.y - b.y));
  const pivot = points[0];

  // Then sort all remaining points based on the angle between the pivot and itself
  const hull = points.slice(1).sort((a, b) => polarAngle(a, pivot) - polarAngle(b, pivot));

  // The pivot is always on the hull
  hull.unshift(pivot);

  let n = hull.length;
  let m = 1;
  for (let i = 2; i < n; i++) {
    while (ccw(hull[m - 1], hull[m], hull[i]) <= 0) {
      if (m > 1) {
        m -= 1;
      } else if (m === i) {
        break;
      } else {
        i += 1;
      }
    }

    m += 1;
    [hull[i], hull[m]] = [hull[m], hull[i]];
  }

  return hull.slice(0, m + 1);
} 
```

```
def graham_scan(gift):
    gift = list(set(gift))  # Remove duplicate points
    start = min(gift, key=lambda p: (p[1], p[0]))  # Must be in hull
    gift.remove(start)

    s = sorted(gift, key=lambda point: polar_angle(start, point))
    hull = [start, s[0], s[1]]

    # Remove points from hull that make the hull concave
    for pt in s[2:]:
        while not counter_clockwise(hull[-2], hull[-1], pt):
            del hull[-1]
        hull.append(pt)

    return hull 
```

```
func grahamScan(points []point) []point {
    sort.Slice(points, func(a, b int) bool {
        return points[a].y < points[b].y || (points[a].y == points[b].y && points[a].x < points[b].x)
    })

    start := points[0]
    points = points[1:]

    sort.Slice(points, func(a, b int) bool {
        return polarAngle(start, points[a]) < polarAngle(start, points[b])
    })

    hull := []point{start, points[0], points[1]}
    for _, p := range points[2:] {
        for !counterClockwise(hull[len(hull)-2], hull[len(hull)-1], p) {
            hull = hull[:len(hull)-1]
        }
        hull = append(hull, p)
    }

    return hull
} 
```

```
static List<Point> grahamScan(List<Point> gift) {
    gift = gift.stream()
               .distinct()
               .sorted(Comparator.comparingDouble(point -> -point.y))
               .collect(Collectors.toList());

    Point pivot = gift.get(0);

    // Sort the remaining Points based on the angle between the pivot and itself
    List<Point> hull = gift.subList(1, gift.size());
    hull.sort(Comparator.comparingDouble(point -> polarAngle(point, pivot)));

    // The pivot is always on the hull
    hull.add(0, pivot);

    int n = hull.size();
    int m = 1;

    for (int i = 2; i < n; i++) {
        while (ccw(hull.get(m - 1), hull.get(m), hull.get(i)) <= 0) {
            if (m > 1) {
                m--;
            } else if (m == 1) {
                break;
            } else {
                i++;
            }
        }
        m++;

        Point temp = hull.get(i);
        hull.set(i, hull.get(m));
        hull.set(m, temp);
    }
    return hull.subList(0, m + 1);
} 
```

```
(defun atan2 (y x)
  "Calculates the angle of a point in the euclidean plane in radians"
  (cond
    ((> x 0)                    (atan y x))
    ((and (< x 0) (>= y 0))     (+ (atan y x) pi))
    ((and (< x 0) (< y 0))      (- (atan y x) pi))
    ((and (eql x 0) (> y 0))    (/ pi 2))
    ((and (eql x 0) (< y 0))    (- (/ pi 2)))
    ;; The -1 signifies an exception and is usefull later for sorting by the polar angle
    ((and (eql x 0) (eql y 0))  -1)))

(defun polar-angle (ref point)
  "Returns the polar angle from a point relative to a reference point"
  (atan2 (- (point-y point) (point-y ref)) (- (point-x point) (point-x ref))))

(defun lowest-point (gift)
  "Returns the lowest point of a gift"
  (reduce
    (lambda (p1 p2)
      (if (< (point-y p1) (point-y p2)) p1 p2))
    gift))

(defun graham-scan (gift)
  "Finds the convex hull of a distribution of points with a graham scan"
  ;; An empty list evaluates to false (nil) and a non-empty list evaluates to true (t).
  ;; We can therefore use 'gift' instead of '(> (length gift) 0)'.
  (if gift
      (labels ((wrap (sorted-points hull)
                 (if sorted-points
                   ;; This covers the case where the hull has one or more element.
                   ;; We aren't concerned about the hull being empty, because then the gift must
                   ;; also be empty and this function is never given an empty gift.
                     (if (rest hull)
                         (if (<= (ccw (first sorted-points) (first hull) (second hull)) 0)
                             (wrap sorted-points (rest hull))
                             (wrap (rest sorted-points) (cons (first sorted-points) hull)))
                         (wrap (rest sorted-points) (list (first sorted-points) (first hull))))
                     hull)))
        ;; Because 'sort' shuffles things around destructively, graham-scan is also destructive. But
        ;; since the order of the points is generally not important, this shouldn't cause a problem.
        (let* ((lowest (lowest-point gift))
                (sorted (sort gift #'< :key (lambda (p) (polar-angle lowest p)))))
          (wrap sorted (list lowest))))
      nil)) 
```

```
std::vector<point> graham_scan(std::vector<point>& points) {
  // selecting lowest point as pivot
  size_t low_index = 0;
  for (size_t i = 1; i < points.size(); i++) {
    if (points[i].y < points[low_index].y) {
      low_index = i;
    }
  }
  std::swap(points[0], points[low_index]);
  point pivot = points[0];

  // sorting points by polar angle
  std::sort(
      points.begin() + 1,
      points.end(),
      &pivot {
        return polar_angle(pivot, pa) < polar_angle(pivot, pb);
      });

  // creating convex hull
  size_t m = 1;
  for (size_t i = 2; i < points.size(); i++) {
    while (ccw(points[m - 1], points[m], points[i]) <= 0) {
      if (m > 1) {
        m--;
        continue;
      } else if (i == points.size()) {
        break;
      } else {
        i++;
      }
    }
    m++;
    std::swap(points[i], points[m]);
  }
  return std::vector<point>(points.begin(), points.begin() + m + 1);
} 
```

```
def graham_scan(gift):
    gift = list(set(gift)) # Remove the duplicate points if any.
    start = min(gift, key=(p -> (p.x, p.y)))
    gift.remove(start)

    s = sorted(gift, key=(point -> start.angle(point)))
    hull = [start, s[0], s[1]]

    # Remove the hull points that make the hull concave
    for point in s[2:]:
        while not counter_clockwise(hull[-2], hull[-1], point):
            del hull[-1]
        hull.append(point)
    return hull 
```

```
fn graham_scan(mut points: Vec<Point>) -> Vec<Point> {
    if points.is_empty() {
        return Vec::new();
    }

    // Unwrap is safe because length is > 0
    let start = *points.iter().min().unwrap();
    points.retain(|a| a != &start);
    points.sort_unstable_by(|a, b| polar_angle(&start, a).partial_cmp(&polar_angle(&start, b)).unwrap());

    let mut hull: Vec<Point> = vec![start, points[0], points[1]];

    for pt in points[2..points.len()].iter() {
        while counter_clockwise(&hull[hull.len() - 2], &hull[hull.len() - 1], pt) < 0.0 {
            hull.pop();
        }
        hull.push(*pt);
    }
    hull
} 
```

### 参考文献

1.Graham, Ronald L, 一种确定有限平面集合凸包的高效算法，*Elsevier*，1972。

## 示例代码

```
struct Point
    x::Float64
    y::Float64
end

function ccw(a::Point, b::Point, c::Point)
    return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x))
end

function graham_scan!(points::Vector{Point})
    N = length(points)

    # Place the lowest point at the start of the array
    sort!(points, by = item -> item.y)

    # Sort all other points according to angle with that point
    other_points = sort(points[2:end], by = item -> atan(item.y - points[1].y,
                                                         item.x - points[1].x))

    # Place points sorted by angle back into points vector
    for i in 1:length(other_points)
        points[i+1] = other_points[i]
    end

    # M will be the point on the hull
    M = 2
    for i = 1:N
        while (ccw(points[M-1], points[M], points[i]) <= 0)
            if (M > 2)
                M -= 1
            # All points are collinear
            elseif (i == N)
                break
            else
                i += 1
            end
        end

        # ccw point found, updating hull and swapping points
        M += 1
        points[i], points[M] = points[M], points[i]
    end

    return points[1:M]
end

function main()
    # This hull is just a simple test so we know what the output should be
    points = [
        Point(-5,2), Point(5,7), Point(-6,-12), Point(-14,-14), Point(9,9),
        Point(-1,-1), Point(-10,11), Point(-6,15), Point(-6,-8), Point(15,-9),
        Point(7,-7), Point(-2,-9), Point(6,-5), Point(0,14), Point(2,8)
    ]
    hull = graham_scan!(points)
    println(hull)
end

main() 
```

```
import Data.List (sortOn, minimumBy)
import Data.Function (on)

type Point = (Double, Double)

ccw :: Point -> Point -> Point -> Double
ccw (xa, ya) (xb, yb) (xc, yc) = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)

grahamScan :: [Point] -> [Point]
grahamScan [] = []
grahamScan pts = wrap sortedPts [p0]
  where p0@(x, y)= minimumBy (compare `on` snd) pts
        sortedPts = sortOn (\(px, py) -> atan2 (py-y) (px-x) ) $ filter (/=p0) pts
        wrap [] ps = ps
        wrap (s:ss) [p] = wrap ss [s, p]
        wrap (s:ss) (p1:p2:ps)
          | ccw s p1 p2 > 0 = wrap (s:ss) (p2:ps)
          | otherwise       = wrap ss (s:p1:p2:ps)

main = do
  -- We build the set of points of integer coordinates within a circle of radius 5
  let pts = [(x,y) | x<-[-5..5], y<-[-5..5], x²+y²<=5²]
  -- And extract the convex hull
  print $ grahamScan pts 
```

```
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct point {
    double x, y;
};

int cmp_points(const void *a, const void *b) {
    struct point* pa = (struct point*) a;
    struct point* pb = (struct point*) b;

    if (pa->y > pb->y) {
        return 1;
    } else if (pa->y < pb->y) {
        return -1;
    } else {
        return 0;
    }
}

double ccw(struct point a, struct point b, struct point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double polar_angle(struct point origin, struct point p) {
    return atan2(p.y - origin.y, p.x - origin.x);
}

void polar_angles_sort(struct point *points, struct point origin, size_t size) {
    if (size < 2) {
        return;
    }

    double pivot_angle = polar_angle(origin, points[size / 2]);

    size_t i = 0;
    size_t j = size - 1;
    while (1) {
        while (polar_angle(origin, points[i]) < pivot_angle) {
            i++;
        }
        while (polar_angle(origin, points[j]) > pivot_angle) {
            j--;
        }

        if (i >= j) {
            break;
        }

        struct point tmp = points[i];
        points[i] = points[j];
        points[j] = tmp;

        i++;
        j--;
    }

    polar_angles_sort(points, origin, i);
    polar_angles_sort(points + i, origin, size - i);
}

size_t graham_scan(struct point *points, size_t size) {
    qsort(points, size, sizeof(struct point), cmp_points);
    polar_angles_sort(points, points[0], size);

    struct point tmp_points[size + 1];
    memcpy(tmp_points + 1, points, size * sizeof(struct point));
    tmp_points[0] = tmp_points[size];

    size_t m = 1;
    for (size_t i = 2; i <= size; ++i) {
        while (ccw(tmp_points[m - 1], tmp_points[m], tmp_points[i]) <= 0) {
            if (m > 1) {
                m--;
                continue;
            } else if (i == size) {
                break;
            } else {
                i++;
            }
        }

        m++;
        struct point tmp = tmp_points[i];
        tmp_points[i] = tmp_points[m];
        tmp_points[m] = tmp;
    }

    memcpy(points, tmp_points + 1, size * sizeof(struct point));

    return m;
}

int main() {
    struct point points[] = {{-5, 2}, {5, 7}, {-6, -12}, {-14, -14}, {9, 9},
                             {-1, -1}, {-10, 11}, {-6, 15}, {-6, -8}, {15, -9},
                             {7, -7}, {-2, -9}, {6, -5}, {0, 14}, {2, 8}};
    size_t num_initial_points = 15;

    printf("Points:\n");
    for (size_t i = 0; i < num_initial_points; ++i) {
        printf("(%f,%f)\n", points[i].x, points[i].y);
    }

    size_t hull_size = graham_scan(points, num_initial_points);

    printf("\nHull:\n");
    for (size_t i = 0; i < hull_size; ++i) {
        printf("(%f,%f)\n", points[i].x, points[i].y);
    }

    return 0;
} 
```

```
function grahamScan(points) {
  // First, sort the points so the one with the lowest y-coordinate comes first (the pivot)
  points = [...points].sort((a, b) => (a.y - b.y));
  const pivot = points[0];

  // Then sort all remaining points based on the angle between the pivot and itself
  const hull = points.slice(1).sort((a, b) => polarAngle(a, pivot) - polarAngle(b, pivot));

  // The pivot is always on the hull
  hull.unshift(pivot);

  let n = hull.length;
  let m = 1;
  for (let i = 2; i < n; i++) {
    while (ccw(hull[m - 1], hull[m], hull[i]) <= 0) {
      if (m > 1) {
        m -= 1;
      } else if (m === i) {
        break;
      } else {
        i += 1;
      }
    }

    m += 1;
    [hull[i], hull[m]] = [hull[m], hull[i]];
  }

  return hull.slice(0, m + 1);
}

function polarAngle(a, b) {
  return Math.atan2(a.y - b.y, a.x - b.x);
}

function ccw(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
}

const points = [
  { x: -5, y: 2 },
  { x: 5, y: 7 },
  { x: -6, y: -12 },
  { x: -14, y: -14 },
  { x: 9, y: 9 },
  { x: -1, y: -1 },
  { x: -10, y: 11 },
  { x: -6, y: 15 },
  { x: -6, y: -8 },
  { x: 15, y: -9 },
  { x: 7, y: -7 },
  { x: -2, y: -9 },
  { x: 6, y: -5 },
  { x: 0, y: 14 },
  { x: 2, y: 8 },
];

const convexHull = grahamScan(points);
console.log("The points in the hull are:");
convexHull.forEach(p => console.log(`(${p.x}, ${p.y})`)); 
```

```
from math import atan2

def counter_clockwise(p1, p2, p3):
    """Is the turn counter-clockwise?"""
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) >= (p2[1] - p1[1]) * (p3[0] - p1[0])

def polar_angle(ref, point):
    """Find the polar angle of a point relative to a reference point"""
    return atan2(point[1] - ref[1], point[0] - ref[0])

def graham_scan(gift):
    gift = list(set(gift))  # Remove duplicate points
    start = min(gift, key=lambda p: (p[1], p[0]))  # Must be in hull
    gift.remove(start)

    s = sorted(gift, key=lambda point: polar_angle(start, point))
    hull = [start, s[0], s[1]]

    # Remove points from hull that make the hull concave
    for pt in s[2:]:
        while not counter_clockwise(hull[-2], hull[-1], pt):
            del hull[-1]
        hull.append(pt)

    return hull

def main():
    test_gift = [
        (-5, 2),
        (5, 7),
        (-6, -12),
        (-14, -14),
        (9, 9),
        (-1, -1),
        (-10, 11),
        (-6, 15),
        (-6, -8),
        (15, -9),
        (7, -7),
        (-2, -9),
        (6, -5),
        (0, 14),
        (2, 8),
    ]
    hull = graham_scan(test_gift)

    print("The points in the hull are:")
    for point in hull:
        print(point)

main() 
```

```
package main

import (
    "fmt"
    "math"
    "sort"
)

type point struct {
    x, y int
}

func counterClockwise(p1, p2, p3 point) bool {
    return (p3.y-p1.y)*(p2.x-p1.x) >= (p2.y-p1.y)*(p3.x-p1.x)
}

func polarAngle(ref, point point) float64 {
    return math.Atan2(float64(point.y-ref.y), float64(point.x-ref.x))
}

func grahamScan(points []point) []point {
    sort.Slice(points, func(a, b int) bool {
        return points[a].y < points[b].y || (points[a].y == points[b].y && points[a].x < points[b].x)
    })

    start := points[0]
    points = points[1:]

    sort.Slice(points, func(a, b int) bool {
        return polarAngle(start, points[a]) < polarAngle(start, points[b])
    })

    hull := []point{start, points[0], points[1]}
    for _, p := range points[2:] {
        for !counterClockwise(hull[len(hull)-2], hull[len(hull)-1], p) {
            hull = hull[:len(hull)-1]
        }
        hull = append(hull, p)
    }

    return hull
}

func main() {
    points := []point{{-5, 2}, {5, 7}, {-6, -12}, {-14, -14}, {9, 9},
        {-1, -1}, {-10, 11}, {-6, 15}, {-6, -8}, {15, -9},
        {7, -7}, {-2, -9}, {6, -5}, {0, 14}, {2, 8}}

    fmt.Println("The points in the hull are:")
    hull := grahamScan(points)
    for _, p := range hull {
        fmt.Println(p)
    }
} 
```

```
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class GrahamScan {

    static class Point {
        public double x;
        public double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object o) {
            if (o == null) return false;
            if (o == this) return true;
            if (!(o instanceof Point)) return false;
            Point p = (Point)o;
            return p.x == this.x && p.y == this.y;
        }
    }

    static double ccw(Point a, Point b, Point c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }

    static double polarAngle(Point origin, Point p) {
        return Math.atan2(p.y - origin.y, p.x - origin.x);
    }

    static List<Point> grahamScan(List<Point> gift) {
        gift = gift.stream()
                   .distinct()
                   .sorted(Comparator.comparingDouble(point -> -point.y))
                   .collect(Collectors.toList());

        Point pivot = gift.get(0);

        // Sort the remaining Points based on the angle between the pivot and itself
        List<Point> hull = gift.subList(1, gift.size());
        hull.sort(Comparator.comparingDouble(point -> polarAngle(point, pivot)));

        // The pivot is always on the hull
        hull.add(0, pivot);

        int n = hull.size();
        int m = 1;

        for (int i = 2; i < n; i++) {
            while (ccw(hull.get(m - 1), hull.get(m), hull.get(i)) <= 0) {
                if (m > 1) {
                    m--;
                } else if (m == 1) {
                    break;
                } else {
                    i++;
                }
            }
            m++;

            Point temp = hull.get(i);
            hull.set(i, hull.get(m));
            hull.set(m, temp);
        }
        return hull.subList(0, m + 1);
    }

    public static void main(String[] args) {
        ArrayList<Point> points = new ArrayList<>();

        points.add(new Point(-5, 2));
        points.add(new Point(5, 7));
        points.add(new Point(-6, -12));
        points.add(new Point(-14, -14));
        points.add(new Point(9, 9));
        points.add(new Point(-1, -1));
        points.add(new Point(-10, 11));
        points.add(new Point(-6, 15));
        points.add(new Point(-6, -8));
        points.add(new Point(15, -9));
        points.add(new Point(7, -7));
        points.add(new Point(-2, -9));
        points.add(new Point(6, -5));
        points.add(new Point(0, 14));
        points.add(new Point(2, 8));

        List<Point> convexHull = grahamScan(points);

        convexHull.forEach(p -> System.out.printf("% 1.0f, % 1.0f\n", p.x, p.y));
    }
} 
```

```
;;;; Graham scan implementation in Common Lisp

(defstruct (point (:constructor make-point (x y))) x y)

(defun ccw (p1 p2 p3)
  "Determines if a turn between three points is counterclockwise"
  (-
    (*
      (- (point-y p2) (point-y p1))
      (- (point-x p3) (point-x p1)))
    (*
      (- (point-y p3) (point-y p1))
      (- (point-x p2) (point-x p1)))))

(defun atan2 (y x)
  "Calculates the angle of a point in the euclidean plane in radians"
  (cond
    ((> x 0)                    (atan y x))
    ((and (< x 0) (>= y 0))     (+ (atan y x) pi))
    ((and (< x 0) (< y 0))      (- (atan y x) pi))
    ((and (eql x 0) (> y 0))    (/ pi 2))
    ((and (eql x 0) (< y 0))    (- (/ pi 2)))
    ;; The -1 signifies an exception and is usefull later for sorting by the polar angle
    ((and (eql x 0) (eql y 0))  -1)))

(defun polar-angle (ref point)
  "Returns the polar angle from a point relative to a reference point"
  (atan2 (- (point-y point) (point-y ref)) (- (point-x point) (point-x ref))))

(defun lowest-point (gift)
  "Returns the lowest point of a gift"
  (reduce
    (lambda (p1 p2)
      (if (< (point-y p1) (point-y p2)) p1 p2))
    gift))

(defun graham-scan (gift)
  "Finds the convex hull of a distribution of points with a graham scan"
  ;; An empty list evaluates to false (nil) and a non-empty list evaluates to true (t).
  ;; We can therefore use 'gift' instead of '(> (length gift) 0)'.
  (if gift
      (labels ((wrap (sorted-points hull)
                 (if sorted-points
                   ;; This covers the case where the hull has one or more element.
                   ;; We aren't concerned about the hull being empty, because then the gift must
                   ;; also be empty and this function is never given an empty gift.
                     (if (rest hull)
                         (if (<= (ccw (first sorted-points) (first hull) (second hull)) 0)
                             (wrap sorted-points (rest hull))
                             (wrap (rest sorted-points) (cons (first sorted-points) hull)))
                         (wrap (rest sorted-points) (list (first sorted-points) (first hull))))
                     hull)))
        ;; Because 'sort' shuffles things around destructively, graham-scan is also destructive. But
        ;; since the order of the points is generally not important, this shouldn't cause a problem.
        (let* ((lowest (lowest-point gift))
                (sorted (sort gift #'< :key (lambda (p) (polar-angle lowest p)))))
          (wrap sorted (list lowest))))
      nil))

(defvar gift
  (map
    'list
    (lambda (e) (apply #'make-point e))
    '((-5 2) (5 7) (-6 -12) (-14 -14) (9 9)
      (-1 -1) (-10 11) (-6 15) (-6 -8) (15 -9)
      (7 -7) (-2 -9) (6 -5) (0 14) (2 8))))

;; This should print out the following:
;; (#S(POINT :X -10 :Y 11) #S(POINT :X -6 :Y 15) #S(POINT :X 0 :Y 14)
;; #S(POINT :X 9 :Y 9) #S(POINT :X 7 :Y -7) #S(POINT :X -6 :Y -12))
(print (graham-scan gift)) 
```

```
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

struct point {
  double x;
  double y;
};

std::ostream& operator<<(std::ostream& os, const std::vector<point>& points) {
  for (auto p : points) {
    os << "(" << p.x << ", " << p.y << ")\n";
  }
  return os;
}

double ccw(const point& a, const point& b, const point& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

double polar_angle(const point& origin, const point& p) {
  return std::atan2(p.y - origin.y, p.x - origin.x);
}

std::vector<point> graham_scan(std::vector<point>& points) {
  // selecting lowest point as pivot
  size_t low_index = 0;
  for (size_t i = 1; i < points.size(); i++) {
    if (points[i].y < points[low_index].y) {
      low_index = i;
    }
  }
  std::swap(points[0], points[low_index]);
  point pivot = points[0];

  // sorting points by polar angle
  std::sort(
      points.begin() + 1,
      points.end(),
      &pivot {
        return polar_angle(pivot, pa) < polar_angle(pivot, pb);
      });

  // creating convex hull
  size_t m = 1;
  for (size_t i = 2; i < points.size(); i++) {
    while (ccw(points[m - 1], points[m], points[i]) <= 0) {
      if (m > 1) {
        m--;
        continue;
      } else if (i == points.size()) {
        break;
      } else {
        i++;
      }
    }
    m++;
    std::swap(points[i], points[m]);
  }
  return std::vector<point>(points.begin(), points.begin() + m + 1);
}

int main() {
  std::vector<point> points = {{-5, 2},
                               {5, 7},
                               {-6, -12},
                               {-14, -14},
                               {9, 9},
                               {-1, -1},
                               {-10, 11},
                               {-6, 15},
                               {-6, -8},
                               {15, -9},
                               {7, -7},
                               {-2, -9},
                               {6, -5},
                               {0, 14},
                               {2, 8}};
  std::cout << "original points are as follows:\n" << points;
  const std::vector<point> hull = graham_scan(points);
  std::cout << "points in hull are as follows:\n" << hull;
  return 0;
} 
```

```
from math import atan2

data point(x=0, y=0):
    def angle(self, other):
        """Computes the angle between the two points"""
        match point(x1, y1) in other:
            return atan2(y1 - self.y, x1 - self.x)
    def __str__(self):
        return f'({self.x}, {self.y})'

# Is the turn counter-clockwise?
def counter_clockwise(p1, p2, p3) =
    (p3.y - p1.y) * (p2.x - p1.x) >= (p2.y - p1.y) * (p3.x - p1.x)

def graham_scan(gift):
    gift = list(set(gift)) # Remove the duplicate points if any.
    start = min(gift, key=(p -> (p.x, p.y)))
    gift.remove(start)

    s = sorted(gift, key=(point -> start.angle(point)))
    hull = [start, s[0], s[1]]

    # Remove the hull points that make the hull concave
    for point in s[2:]:
        while not counter_clockwise(hull[-2], hull[-1], point):
            del hull[-1]
        hull.append(point)
    return hull

if __name__ == '__main__':
    test_gift = [
            (-5, 2),
            (5, 7),
            (-6, -12),
            (-14, -14),
            (9, 9),
            (-1, -1),
            (-10, 11),
            (-6, 15),
            (-6, -8),
            (15, -9),
            (7, -7),
            (-2, -9),
            (6, -5),
            (0, 14),
            (2, 8),
    ] |> map$(p -> point(*p)) |> list
    hull = graham_scan(test_gift)
    "The points in the hull are:" |> print
    "\n".join(map(str, hull)) |> print 
```

```
use std::cmp::Ordering;

#[derive(Debug, PartialEq, Copy, Clone)]
struct Point {
    x: f64,
    y: f64,
}

impl Eq for Point {}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.y == other.y {
            self.x.partial_cmp(&other.x)
        } else {
            self.y.partial_cmp(&other.y)
        }
    }
}

// Defines an order for Points so they can be sorted
impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        // Neither field of Point will be NaN, so this is safe
        self.partial_cmp(other).unwrap()
    }
}

// Determines whether the angle abc is clockwise, counter-clockwise or colinear
// result > 0 : counter-clockwise
// result = 0 : colinear
// result < 0 : clockwise
fn counter_clockwise(a: &Point, b: &Point, c: &Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

// Calculate the polar angle of a  point relative to a reference point.
fn polar_angle(reference: &Point, point: &Point) -> f64 {
    (point.y - reference.y).atan2(point.x - reference.x)
}

fn graham_scan(mut points: Vec<Point>) -> Vec<Point> {
    if points.is_empty() {
        return Vec::new();
    }

    // Unwrap is safe because length is > 0
    let start = *points.iter().min().unwrap();
    points.retain(|a| a != &start);
    points.sort_unstable_by(|a, b| polar_angle(&start, a).partial_cmp(&polar_angle(&start, b)).unwrap());

    let mut hull: Vec<Point> = vec![start, points[0], points[1]];

    for pt in points[2..points.len()].iter() {
        while counter_clockwise(&hull[hull.len() - 2], &hull[hull.len() - 1], pt) < 0.0 {
            hull.pop();
        }
        hull.push(*pt);
    }
    hull
}

fn main() {
    let points = vec![
        Point { x:  -5.0, y:   2.0 },
        Point { x:   5.0, y:   7.0 },
        Point { x:  -6.0, y: -12.0 },
        Point { x: -14.0, y: -14.0 },
        Point { x:   9.0, y:   9.0 },
        Point { x:  -1.0, y:  -1.0 },
        Point { x: -10.0, y:  11.0 },
        Point { x:  -6.0, y:  15.0 },
        Point { x:  -6.0, y:  -8.0 },
        Point { x:  15.0, y:  -9.0 },
        Point { x:   7.0, y:  -7.0 },
        Point { x:  -2.0, y:  -9.0 },
        Point { x:   6.0, y:  -5.0 },
        Point { x:   0.0, y:  14.0 },
        Point { x:   2.0, y:   8.0 },
    ];

    let hull_points = graham_scan(points);
    println!("{:#?}", hull_points);
} 
```

## 许可证

##### 代码示例

代码示例受 MIT 许可协议（见[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)）许可。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并受[Creative Commons Attribution-ShareAlike 4.0 国际许可协议](https://creativecommons.org/licenses/by-sa/4.0/legalcode)许可。

[(https://creativecommons.org/licenses/by-sa/4.0/)]

![图片](img/c7782818305d7cc32e33f74558a972b7.png)[(https://creativecommons.org/licenses/by-sa/4.0/)]

##### 提交的请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下提交请求已修改本章的文本或图形：

+   无
