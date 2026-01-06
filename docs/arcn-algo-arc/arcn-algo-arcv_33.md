# Jarvis March

> 原文：[`www.algorithm-archive.org/contents/jarvis_march/jarvis_march.html`](https://www.algorithm-archive.org/contents/jarvis_march/jarvis_march.html)

第一维二维凸包算法最初由 R. A. Jarvis 于 1973 年开发 [[1]](#cite-1)。尽管存在其他凸包算法，但这个算法通常被称为*礼物包装算法*。

该算法背后的思想很简单。如果我们从一个随机的点分布开始，我们可以通过首先从最左边的点开始，并使用原点来计算模拟中每个其他点之间的角度来找到凸包。作为备注，"角度"可以用叉积或点积来大致近似，这在某些实现中很常见。具有最大内角的那一点被选为凸包中的下一个点，我们在这两点之间画一条线。从那里，我们使用已知的两个点再次计算模拟中所有其他点之间的角度。然后我们选择具有最大内角的那一点，并将模拟向前推进。我们重复这个过程，直到回到我们的原始点。在这个模拟中选择的点集将是凸包。

如我们所预期，这个算法并不特别高效，其运行时间为 ，其中  是点的数量，  是凸包的大小。作为备注，Jarvis March 可以推广到更高维度。自那时以来，已经有许多其他算法推动了二维礼物包装算法领域的发展，包括 Graham 扫描和 Chan 算法，这些将在适当的时候讨论。

### 参考文献列表

1. Jarvis, Ray A, On the identification of the convex hull of a finite set of points in the plane, *Elsevier*, 1973.

## 示例代码

##### JarvisMarch.cs

```
// submitted by Julian Schacher (jspp) with great help by gustorn
using System;
using System.Collections.Generic;
using System.Linq;

namespace JarvisMarch
{
    public struct Vector
    {
        public readonly int x;
        public readonly int y;

        public Vector(int xValue, int yValue)
        {
            this.x = xValue;
            this.y = yValue;
        }

        public override bool Equals(object obj) => obj is Vector v && this.x == v.x && this.y == v.y;
        public override int GetHashCode() => (17 * 23 + this.x) * 23 + this.y;

        public static bool operator==(Vector a, Vector b) => a.Equals(b);
        public static bool operator!=(Vector a, Vector b) => !(a == b);
    }

    public class JarvisMarch
    {
        public List<Vector> Run(List<Vector> points)
        {
            var convexHull = new List<Vector>();

            // Set the intial pointOnHull to the point of the list, where the x-position is the lowest.
            var pointOnHull = points.Aggregate((leftmost, current) => leftmost.x < current.x ? leftmost : current);

            // Continue searching for the next pointOnHull until the next pointOnHull is equal to the first point of the convex hull.
            do
            {
                convexHull.Add(pointOnHull);

                // Search for the next pointOnHull by looking which of the points is the next most outer point.
                pointOnHull = points.Aggregate((potentialNextPointOnHull, current) =>
                {
                    // Returns true, if potentialNextPointOnHull is equal to the current pointOnHull or if the current point is left of the line defined by pointOnHull and potentialNextPointOnHull.
                    if (potentialNextPointOnHull == pointOnHull || IsLeftOf(pointOnHull, potentialNextPointOnHull, current))
                        return current;
                    return potentialNextPointOnHull;
                });

                // Check if the gift wrap is completed.
            } while (pointOnHull != convexHull[0]);

            return convexHull;
        }

        // Returns true, if p is left of the line defined by a and b.
        private bool IsLeftOf(Vector a, Vector b, Vector p) => (b.x - a.x) * (p.y - a.y) > (p.x - a.x) * (b.y - a.y);
    }
} 
```

##### Program.cs

```
// submitted by Julian Schacher (jspp) with great help by gustorn
using System;
using System.Collections.Generic;

namespace JarvisMarch
{
    class Program
    {
        static void Main(string[] args)
        {
            System.Console.WriteLine("JarvisMarch");
            // Example list of points.
            // The points are represented by vectors here, but that doesn't really matter.
            var points = new List<Vector>()
            {
                new Vector(-5, 2),
                new Vector(5, 7),
                new Vector(-6, -12),
                new Vector(-14, -14),
                new Vector(9, 9),
                new Vector(-1, -1),
                new Vector(-10, 11),
                new Vector(-6, 15),
                new Vector(-6, -8),
                new Vector(15, -9),
                new Vector(7, -7),
                new Vector(-2, -9),
                new Vector(6, -5),
                new Vector(0, 14),
                new Vector(2, 8),
            };
            var jarvisMarch = new JarvisMarch();
            var giftWrap = jarvisMarch.Run(points);

            // Print the points of the gift wrap.
            foreach (var point in giftWrap)
                System.Console.WriteLine($"{point.x}, {point.y}");
        }
    }
} 
```

```
struct Pos
    x::Float64
    y::Float64
end

function jarvis_cross(point1::Pos, point2::Pos, point3::Pos)
    vec1 = Pos(point2.x - point1.x, point2.y - point1.y)
    vec2 = Pos(point3.x - point2.x, point3.y - point2.y)
    ret_cross = vec1.x*vec2.y - vec1.y*vec2.x
    return ret_cross*ret_cross
end

function jarvis_march(points::Vector{Pos})
    hull = Vector{Pos}()

    # sorting array based on leftmost point
    sort!(points, by = item -> item.x)
    push!(hull, points[1])

    i = 1
    curr_point = points[2]

    # Find cross product between points
    curr_product = jarvis_cross(Pos(0,0), hull[1], curr_point)
    while (curr_point != hull[1])
        for point in points
                product = 0.0
            if (i == 1)
                if (hull[i] != point)
                    product = jarvis_cross(Pos(0,0), hull[i], point)
                end
            else
                if (hull[i] != point && hull[i-1] != point)
                    product = jarvis_cross(hull[i-1], hull[i], point)
                end
            end
            if (product > curr_product)
                curr_point = point
                curr_product = product
            end
        end
        push!(hull, curr_point)
        curr_product = 0
        i += 1
    end

    return hull
end

function main()

    points = [Pos(2,1.5), Pos(1, 1), Pos(2, 4), Pos(3, 1)]
    hull = jarvis_march(points)
    println(hull)
end

main() 
```

```
import Data.List (sort, maximumBy)
import Data.Function (on)

type Point = (Double, Double)

angle :: Point -> Point -> Point -> Double
angle a@(xa, ya) b@(xb, yb) c@(xc, yc)
  | a==b || c==b = 0
  | theta<0      = theta+2*pi
  | otherwise    = theta
  where thetaA = atan2 (ya-yb) (xa-xb)
        thetaC = atan2 (yc-yb) (xc-xb)
        theta = thetaC - thetaA

jarvisMarch :: [Point] -> [Point]
jarvisMarch [] = []
jarvisMarch pts = p0 : wrap (x, y-1) p0
  where p0@(x, y)= minimum pts
        wrap p1 p2
          | pm == p0  = []
          | otherwise = pm : wrap p2 pm
          where pm = maximumBy (compare `on` angle p1 p2) pts

main = do
  let pts = filter (\(x,y) -> x²+y²<=5²) [(x,y)|x<-[-5..5], y<-[-5..5]]
  print $ jarvisMarch pts 
```

```
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

struct point {
    double x,y;
};

struct point left_most_point(struct point *points, size_t num_points) {
    struct point ret = points[0];

    for (size_t i = 0; i < num_points; ++i) {
        if (points[i].x < ret.x) {
            ret = points[i];
        } else if(points[i].x == ret.x) {
            if (points[i].y < ret.y) {
                ret = points[i];
            }
        }
    }

    return ret;
}

bool equal(struct point a, struct point b) {
    return a.x == b.x && a.y == b.y;
}

double winding(struct point p, struct point q, struct point r) {
    return (q.x - p.x)*(r.y - p.y) - (q.y - p.y)*(r.x - p.x);
}

size_t jarvis_march(struct point *points, struct point *hull_points,
                    size_t num_points) {
    struct point hull_point = left_most_point(points, num_points);
    struct point end_point;

    size_t i = 0;
    do {
        hull_points[i] = hull_point;
        end_point = points[0];

        for (size_t j = 1; j < num_points; ++j) {
            if (equal(end_point, hull_point) ||
                    winding(hull_points[i], end_point, points[j]) > 0.0) {
                end_point = points[j];
            }
        }

        i++;
        hull_point = end_point;
    } while (!equal(end_point, hull_points[0]));

    return i;
}

int main() {
    struct point points[] = {
        { -5.0, 2.0 },
        { 5.0, 7.0 },
        { -6.0, -12.0 },
        { -14.0, -14.0 },
        { 9.0, 9.0 },
        { -1.0, -1.0 },
        { -10.0, 11.0 },
        { -6.0, 15.0 },
        { -6.0, -8.0 },
        { 15.0, -9.0 },
        { 7.0, -7.0 },
        { -2.0, -9.0 },
        { 6.0, -5.0 },
        { 0.0, 14.0 },
        { 2.0, 8.0 }
    };
    struct point hull_points[15];

    size_t num_hull_points = jarvis_march(points, hull_points, 15);

    printf("The Hull points are:\n");
    for (size_t i = 0; i < num_hull_points; ++i) {
        printf("x=%f y=%f\n", hull_points[i].x, hull_points[i].y);
    }

    return 0;
} 
```

```
function jarvisMarch(points) {
  const hull = [];

  let pointOnHull = points.reduce((leftmost, current) => leftmost.x < current.x ? leftmost : current);
  do {
    hull.push(pointOnHull);
    pointOnHull = points.reduce(chooseNextPointOnHull(pointOnHull));
  } while (pointOnHull !== hull[0]);

  return hull;
}

function chooseNextPointOnHull(currentPoint) {
  return function (nextPoint, candidate) {
      if (nextPoint === currentPoint || isLeftOf({ a: currentPoint, b: nextPoint }, candidate)) {
        return candidate;
      }
      return nextPoint;
  }
}

function isLeftOf({ a, b }, p) {
  return (b.x - a.x) * (p.y - a.y) > (p.x - a.x) * (b.y - a.y);
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
  { x: 2, y: 8 }
];

const convexHull = jarvisMarch(points);
convexHull.forEach(p => console.log(`(${p.x}, ${p.y})`)); 
```

```
# Is the turn counter clockwise?
def ccw(p1, p2, p3):
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) \
        >= (p2[1] - p1[1]) * (p3[0] - p1[0])

def jarvis_march(gift):
    n = len(gift)  # Number of points in list
    point_on_hull = min(gift)  # leftmost point in gift
    hull = [point_on_hull]  # leftmost point guaranteed to be in hull

    while True:
        # Candidate for next point in hull
        endpoint = gift[0]
        for j in range(1, n):
            if endpoint == point_on_hull \
               or not ccw(gift[j], hull[-1], endpoint):
                endpoint = gift[j]

        point_on_hull = endpoint

        # Check if we have completely wrapped gift
        if hull[0] == endpoint:
            break
        else:
            hull.append(point_on_hull)

    return hull

def main():
    test_gift = [
        (-5, 2), (5, 7), (-6, -12), (-14, -14), (9, 9),
        (-1, -1), (-10, 11), (-6, 15), (-6, -8), (15, -9),
        (7, -7), (-2, -9), (6, -5), (0, 14), (2, 8)
    ]
    hull = jarvis_march(test_gift)

    print("The points in the hull are:")
    for point in hull:
        print(point)

if __name__ == "__main__":
    main() 
```

```
#include <vector>
#include <iostream>
#include <algorithm>

struct Point
{
    double x, y;

    bool operator==(const Point& b) const
    {
        return x == b.x && y == b.y;
    }

    bool operator!=(const Point& b) const
    {
        return !(*this == b);
    }
};

std::vector<Point> jarvis_march(const std::vector<Point>& points)
{
    std::vector<Point> hull_points;

    if(points.empty())
        return hull_points;

    // Left most point
    auto first_point_it = std::min_element(points.begin(), points.end(), [](const Point& a, const Point& b){ return a.x < b.x; });

    auto next_point_it = first_point_it;
    do
    {
        hull_points.push_back(*next_point_it);

        const Point& p1 = hull_points.back();

        // Largest internal angle
        next_point_it = std::max_element(
            points.begin(),
            points.end(),
            p1{
                return (p1 == p2) || (p2.x - p1.x) * (p3.y - p1.y) > (p3.x - p1.x) * (p2.y - p1.y);
            }
        );
    }
    while(*next_point_it != *first_point_it);

    return hull_points;
}

int main() {
    std::vector<Point> points = {
        { -5.0, 2.0 },
        { 5.0, 7.0 },
        { -6.0, -12.0 },
        { -14.0, -14.0 },
        { 9.0, 9.0 },
        { -1.0, -1.0 },
        { -10.0, 11.0 },
        { -6.0, 15.0 },
        { -6.0, -8.0 },
        { 15.0, -9.0 },
        { 7.0, -7.0 },
        { -2.0, -9.0 },
        { 6.0, -5.0 },
        { 0.0, 14.0 },
        { 2.0, 8.0 }
    };

    auto hull_points = jarvis_march(points);

    std::cout << "Hull points are:" << std::endl;

    for(const Point& point : hull_points) {
        std::cout << '(' << point.x << ", " << point.y << ')' << std::endl;
    }
} 
```

```
;;;; Jarvis March implementation

(defstruct (point (:constructor make-point (x y))) x y)

(defun is-left-p (p1 p2 p3)
  "Checks if the point p3 is to the left of the line p1 -> p2"
  (>
    (*
      (- (point-y p3) (point-y p1))
      (- (point-x p2) (point-x p1)))
    (*
      (- (point-y p2) (point-y p1))
      (- (point-x p3) (point-x p1)))))

(defun next-point-on-hull (p1 p2 gift)
  "Finds the next point on the convex hull of a gift"
  (if (null gift)
      p2
      (if (is-left-p p1 p2 (first gift))
          (next-point-on-hull p1 (first gift) (rest gift))
          (next-point-on-hull p1 p2 (rest gift)))))

(defun leftmost-point (gift)
  "Returns the lefmost point of a gift"
  (reduce 
    (lambda (p1 p2)
      (if (< (point-x p1) (point-x p2)) p1 p2))
    gift))

(defun jarvis-march (gift)
  "finds the convex hull of any distribution of points"
  ;deals with the edge cases
  (if (< (length gift) 3)
    gift
    (loop
      with start = (leftmost-point gift)
      with hull = (list start (make-point (point-x start) (- (point-y start) 1)))
      do 
        (setq hull
          (cons 
            (next-point-on-hull (first hull) (second hull) gift)
            hull))
      until (equalp (first hull) start)
      ;deletes extra points
      finally (return (rest (butlast hull))))))

(defvar gift
  (map 
    'list
    (lambda (e) (apply #'make-point e))
    '((2 1.5) (1 1) (2 4) (3 1))))

(print (jarvis-march gift)) 
```

```
import java.util.*;

public class JarvisMarch {

    static class Point {
        private double x;
        private double y;

        public Point(double a, double b) {
            x = a;
            y = b;
        }

        public double getX() {
            return x;
        }
        public double getY() {
            return y;
        }

        public boolean equals(Point p) {
            if (p.getX() == x && p.getY() == y) {
                return true;
            } else {
                return false;
            }
        }
        public double magnitude() {
            return Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2));
        }
    }

    //find the angle by creating two vectors and then using a property of dot products
    private static double angle(Point a, Point b, Point c) {
        Point ab = new Point(b.getX() - a.getX(), b.getY() - a.getY());
        Point bc = new Point(c.getX() - b.getX(), c.getY() - b.getY());
        return Math.acos(-1 * ((ab.getX() * bc.getX()) + (ab.getY() * bc.getY())) /
                               (ab.magnitude() * bc.magnitude()));
    }

    public static ArrayList<Point> jarvisMarch(ArrayList<Point> arr) {
        ArrayList<Point> hull = new ArrayList<Point>();
        Point pointOnHull = new Point(Double.MAX_VALUE, 0);

        //find leftmost point
        for (Point p: arr) {
            if (p.getX() < pointOnHull.getX()) {
                pointOnHull = p;
            }
        }
        hull.add(pointOnHull);

        //look for the rest of the points on the hull
        Point ref;
        while (true) {
            if (hull.size() == 1) {
                ref = new Point(pointOnHull.getX(), pointOnHull.getY() + 1); //finds a third point to use in calculating the angle
            } else {
                ref = hull.get(hull.size() - 2);
            }
            Point endpoint = arr.get(0); //initial canidate for next point in hull
            for (Point p: arr) {
                if (angle(p, pointOnHull, ref) > angle(endpoint, pointOnHull, ref)) { //found a point that makes a greater angle
                    endpoint = p;
                }
            }
            pointOnHull = endpoint;
            if (pointOnHull.equals(hull.get(0))) { //add next point to hull if not equal to the leftmost point
                break;
            } else {
                hull.add(pointOnHull);
            }
        }
        return hull;
    }

    public static void main(String[] args) {

        //test array setup
        ArrayList<Point> gift = new ArrayList<Point>();
        gift.add(new Point(-5, 2));
        gift.add(new Point(5, 7));
        gift.add(new Point(-6, -12));
        gift.add(new Point(-14, -14));
        gift.add(new Point(9, 9));
        gift.add(new Point(-1, -1));
        gift.add(new Point(-10, 11));
        gift.add(new Point(-6, 15));
        gift.add(new Point(-6, -8));
        gift.add(new Point(15, -9));
        gift.add(new Point(7, -7));
        gift.add(new Point(-2, -9));
        gift.add(new Point(6, -5));
        gift.add(new Point(0, 14));
        gift.add(new Point(2, 8));

        //print initial array of points
        System.out.println("Gift:");
        for (Point p: gift) {
            System.out.println("[" + p.getX() + ", " + p.getY() + "]");
        }

        //find and print the array of points in the hull
        ArrayList<Point> hull = jarvisMarch(gift);
        System.out.println("Wrapping:");
        for (Point p: hull) {
            System.out.println("[" + p.getX() + ", " + p.getY() + "]");
        }
    }

} 
```

```
package main

import (
    "fmt"
)

type point struct {
    x, y float64
}

func leftMostPoint(points []point) point {
    ret := points[0]

    for _, p := range points {
        if (p.x < ret.x) || (p.x == ret.x && p.y < ret.y) {
            ret = p
        }
    }

    return ret
}

func (p point) equal(o point) bool {
    return p.x == o.x && p.y == o.y
}

func counterClockWise(p1, p2, p3 point) bool {
    return (p3.y-p1.y)*(p2.x-p1.x) >= (p2.y-p1.y)*(p3.x-p1.x)
}

func jarvisMarch(points []point) []point {
    hullPoints := make([]point, 0)
    hullPoint := leftMostPoint(points)
    hullPoints = append(hullPoints, hullPoint)

    for {
        endPoint := points[0]

        for _, p := range points[1:] {
            if endPoint.equal(hullPoint) || !counterClockWise(p, hullPoints[len(hullPoints)-1], endPoint) {
                endPoint = p
            }
        }

        hullPoint = endPoint

        if endPoint.equal(hullPoints[0]) {
            break
        }

        hullPoints = append(hullPoints, hullPoint)
    }
    return hullPoints
}

func main() {
    points := []point{{-5, 2}, {5, 7}, {-6, -12}, {-14, -14}, {9, 9},
        {-1, -1}, {-10, 11}, {-6, 15}, {-6, -8}, {15, -9},
        {7, -7}, {-2, -9}, {6, -5}, {0, 14}, {2, 8},
    }

    hullPoints := jarvisMarch(points)
    fmt.Println("The hull points are:")

    for _, p := range hullPoints {
        fmt.Printf("x=%f y=%f\n", p.x, p.y)
    }
} 
```

```
struct Point {
    x int
    y int
}

fn left_most_point(points []Point) Point {
    mut ret := points[0]

    for p in points {
        if (p.x < ret.x) || (p.x == ret.x && p.y < ret.y) {
            ret = p
        }
    }

    return ret
}

fn (p Point) equal(o Point) bool {
    return p.x == o.x && p.y == o.x
}

fn counter_clock_wise(p1, p2, p3 Point) bool {
    return (p3.y-p1.y) * (p2.x-p1.x) >= (p2.y-p1.y) * (p3.x-p1.x)
}

fn jarvis_march(points []Point) []Point {
    mut hull_point := left_most_point(points)
    mut hull_points := [hull_point]

    for {
        mut end_point := points[0]

        for i := 1; i < points.len; i++ {
            if end_point.equal(points[i]) || !counter_clock_wise(points[i], hull_points[hull_points.len-1], end_point) {
                end_point = points[i]
            }
        }

        hull_point = end_point
        if end_point.equal(hull_points[0]) {
            break
        }

        hull_points << hull_point
    }
    return hull_points
}

fn main() {
    points := [
        Point{-5, 2}, Point{5, 7}, Point{-6, -12}, Point{-14, -14}, Point{9, 9},
        Point{-1, -1}, Point{-10, 11}, Point{-6, 15}, Point{-6, -8}, Point{15, -9},
        Point{7, -7}, Point{-2, -9}, Point{6, -5}, Point{0, 14}, Point{2, 8}
    ]

    hull_points := jarvis_march(points)

    println('The hull points are:')
    for p in hull_points {
        println('x=$p.x y=$p.y')
    }
} 
```

```
 type Point = (i64, i64);

// Is the turn counter clockwise?
fn turn_counter_clockwise(p1: Point, p2: Point, p3: Point) -> bool {
    (p3.1 - p1.1) * (p2.0 - p1.0) >= (p2.1 - p1.1) * (p3.0 - p1.0)
}

fn jarvis_march(gift: &[Point]) -> Option<Vec<Point>> {
    // There can only be a convex hull if there are more than 2 points
    if gift.len() < 3 {
        return None;
    }

    let leftmost_point = gift
        // Iterate over all points
        .iter()
        // Find the point with minimum x
        .min_by_key(|i| i.0)
        // If there are no points in the gift, there might
        // not be a minimum. Unwrap fails (panics) the program
        // if there wasn't a minimum, but we know there always
        // is because we checked the size of the gift.
        .unwrap()
        .clone();

    let mut hull = vec![leftmost_point];

    let mut point_on_hull = leftmost_point;
    loop {
        // Search for the next point on the hull
        let mut endpoint = gift[0];
        for i in 1..gift.len() {
            if endpoint == point_on_hull || !turn_counter_clockwise(gift[i], hull[hull.len() - 1], endpoint) {
                endpoint = gift[i];
            }
        }

        point_on_hull = endpoint;

        // Stop whenever we got back to the same point
        // as we started with, and we wrapped the gift
        // completely.
        if hull[0] == endpoint {
            break;
        } else {
            hull.push(point_on_hull);
        }
    }

    Some(hull)
}

fn main() {
    let test_gift = vec![
        (-5, 2), (5, 7), (-6, -12), (-14, -14), (9, 9),
        (-1, -1), (-10, 11), (-6, 15), (-6, -8), (15, -9),
        (7, -7), (-2, -9), (6, -5), (0, 14), (2, 8)
    ];

    let hull = jarvis_march(&test_gift);

    println!("The points in the hull are: {:?}", hull);
} 
```

```
data point(x=0, y=0):
    def __str__(self):
        return f'({self.x}, {self.y})'

# Is the turn counter-clockwise?
def counter_clockwise(point(p1), point(p2), point(p3)) =
    (p3.y - p1.y) * (p2.x - p1.x) >= (p2.y - p1.y) * (p3.x - p1.x)

def jarvis_march(gift: point[]) -> point[]:
    point_on_hull = min(gift) # The leftmost point in the gift
    hull = [point_on_hull] # It is guaranteed it will be on the hull.

    while True:
        # Candidate for the next point in the hull
        endpoint = gift[0]
        for p in gift:
            if (endpoint == point_on_hull
                or not counter_clockwise(p, hull[-1], endpoint)):
                endpoint = p

        point_on_hull = endpoint

        # Check if the gift is completely covered.
        if hull[0] == endpoint:
            return hull
        hull.append(point_on_hull)

if __name__ == '__main__':
    test_gift = [
        (-5, 2), (5, 7), (-6, -12), (-14, -14), (9, 9),
        (-1, -1), (-10, 11), (-6, 15), (-6, -8), (15, -9),
        (7, -7), (-2, -9), (6, -5), (0, 14), (2, 8)
    ] |> map$(t -> point(*t)) |> list
    hull = jarvis_march(test_gift)

    print("[#] The points in the hull are:")
    for point in hull:
        print(point) 
```

## 许可证

##### 代码示例

代码示例授权于 MIT 许可（见[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并授权于[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。

[](https://creativecommons.org/licenses/by-sa/4.0/)

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 拉取请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下拉请求已修改本章的文本或图形：

+   无
