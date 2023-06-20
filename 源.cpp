#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "math.h"
#include<opencv2/imgproc/types_c.h> 
#include "opencv2/imgcodecs/legacy/constants_c.h"
using namespace std;
using namespace cv;

#define PictureMode 1
#define VideoMode 2

double getLineLength(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}
double getLineGradient(Point p1, Point p2) {
    if (p1.x == p2.x || p1.x - p2.x == 0) {
        p1.x = p1.x + 1;
    }
    return (p1.y - p2.y) / (p1.x - p2.x);
}
//处理离群直线
void handlingOutlierLines(vector<Vec4i>& Lines, size_t threshold) {
    double maxLength, minLength;
    maxLength = getLineLength(Point(Lines[0][0], Lines[0][1]), Point(Lines[0][2], Lines[0][3]));
    minLength = getLineLength(Point(Lines[0][0], Lines[0][1]), Point(Lines[0][2], Lines[0][3]));
    cout << "enter handlingOutlierLines" << endl;
    int number = Lines.size();
    while (number > threshold) {
        double mean = 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < number; i++) {
            if (getLineLength(Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3])) > maxLength) {
                maxLength = getLineLength(Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]));
            }
            if (getLineLength(Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3])) < minLength) {
                minLength = getLineLength(Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]));
            }
            sum += getLineGradient(Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]));
        }
        mean = sum / number;
        cout << "mean: " << mean << endl;
        vector<Vec4i>::iterator it = Lines.begin();
        while (it != Lines.end()) {
            Vec4i vec = *it;
            double gradient = getLineGradient(Point(vec[0], vec[1]), Point(vec[2], vec[3]));
            if (fabs(gradient - mean) > fabs(20 * mean)) {
                it = Lines.erase(it);
            }
            else {
                ++it;
            }
        }
        number = Lines.size();
    }
    cout << "maxLength: " << maxLength << " minLength: " << minLength << "lineNumber: " << Lines.size() << endl;
};
//检测空洞
Mat connected_components_stat(Mat& image) {

    // 二值化
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 200, 255, THRESH_BINARY);

    //开运算、闭运算
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    //morphologyEx(binary, binary, MORPH_DILATE, kernel);
    //morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    imshow("binaryImage", binary);

    //计算连通域
    Mat labels = Mat::zeros(image.size(), CV_32S);
    Mat stats, centroids;
    int num_labels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
    //使用不同的颜色标记连通域
    vector<unsigned char> colors(num_labels);
    // 背景颜色
    colors[0] = 0;
    //将白色空洞染成黑色
    for (int i = 1; i < num_labels; i++) {
        Vec2d pt = centroids.at<Vec2d>(i, 0);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area <= 20 || area > 400 || (y < image.rows / 2 && area>100)) {
            colors[i] = 0;
        }
        else {
            colors[i] = 255;
        }
    }

    Mat result = Mat::zeros(image.size(), CV_8U);
    int w = image.cols;
    int h = image.rows;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int label = labels.at<int>(row, col);
            if (label == 0) continue;
            result.at<unsigned char>(row, col) = colors[label];
        }
    }

    imshow("result", result);
    return result;
}

void detectWhiteDottedLine(string imgPath) {
    Mat image = imread(imgPath);
    imshow("image", image);

    Mat result;
    result = connected_components_stat(image);

    //    Mat gaussImage;
    //    GaussianBlur(binaryImage,gaussImage,Size(3,3),0,0);
    //    imshow("gaussImage",gaussImage);
    //
    Mat cannyImage;
    Canny(result, cannyImage, 235, 250);
    imshow("cannyImage", cannyImage);
    waitKey(0);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    //morphologyEx(cannyImage,cannyImage,MORPH_DILATE,kernel);

    Mat OutputImage = imread(imgPath);
    vector<Vec4i> Lines;
    HoughLinesP(cannyImage, Lines, 1, CV_PI / 360, 30, 20, 20);
    cout << "检测到的直线的数量是：" << Lines.size() << endl;
    //handlingOutlierLines(Lines,Lines.size()-1);
    double max;
    Vec4i max1, max2;
    max = getLineLength(Point(Lines[0][0], Lines[0][1]), Point(Lines[0][2], Lines[0][3]));
    max1 = { Lines[0][0],Lines[0][1],Lines[0][2],Lines[0][3] };
    for (size_t i = 0; i < Lines.size(); i++)
    {
        double lineLength = getLineLength(Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]));
        if (lineLength > max) {
            max = lineLength;
            max2 = max1;
            max1 = { Lines[i][0],Lines[i][1],Lines[i][2], Lines[i][3] };
        }
        if (lineLength > 0) {
            line(OutputImage, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]), Scalar(0, 0, 255), 2, 8);
        }
    }
    cout << max1 << " " << max2 << " " << max << endl;
    imshow("OutputImage", OutputImage);
    waitKey(0);
    destroyAllWindows();
}
void detectWhiteDottedLine2(string imgPath) {
    Mat image = imread(imgPath, IMREAD_GRAYSCALE);
//    Rect rect1(0, 0.5*src.rows, src.cols, 0.5 * src.rows);
//    Mat image = src(rect1);
    imshow("image", image);
    waitKey(0);
    Mat binaryImage;
    threshold(image, binaryImage, 220, 250, THRESH_BINARY);
    imshow("binaryImage", binaryImage);

    Mat morphImage;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    morphologyEx(binaryImage, morphImage, MORPH_DILATE, kernel);
    imshow("morphImage", morphImage);
    waitKey(0);
    //
    //    Mat gaussImage;
    //    GaussianBlur(binaryImage,gaussImage,Size(3,3),0,0);
    //    imshow("gaussImage",gaussImage);

    Mat cannyImage;
    Canny(morphImage, cannyImage, 235, 250);
    imshow("cannyImage", cannyImage);

    //计算连通域
    Mat labels = Mat::zeros(image.size(), CV_32S);
    Mat stats, centroids;
    int num_labels = connectedComponentsWithStats(cannyImage, labels, stats, centroids, 8, 4);
    //使用不同的颜色标记连通域
    vector<unsigned char> colors(num_labels);
    // 背景颜色
    colors[0] = 0;
    //将白色空洞染成黑色
    for (int i = 1; i < num_labels; i++) {
        Vec2d pt = centroids.at<Vec2d>(i, 0);
        //int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        //int width = stats.at<int>(i, CC_STAT_WIDTH);
        //int height = stats.at<int>(i, CC_STAT_HEIGHT);
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area <= 35) {
            colors[i] = 0;
        }
        else {
            colors[i] = 255;
        }
    }

    Mat result = Mat::zeros(image.size(), CV_8U);
    int w = image.cols;
    int h = image.rows;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int label = labels.at<int>(row, col);
            if (label == 0) continue;
            result.at<unsigned char>(row, col) = colors[label];
        }
    }

    imshow("result", result);
    //    Mat gaussImage;
    //    GaussianBlur(result,gaussImage,Size(3,3),0,0);
    //    imshow("gaussImage",gaussImage);

    Mat OutputImage = imread(imgPath);
    vector<Vec4i> Lines;
    HoughLinesP(result, Lines, 1, CV_PI / 360, 50, 20, 30);
    cout << "检测到的直线的数量是：" << Lines.size() << endl;
    //handlingOutlierLines(Lines,Lines.size()-1);
    double max;
    Vec4i max1, max2;
    max = getLineLength(Point(Lines[0][0], Lines[0][1]), Point(Lines[0][2], Lines[0][3]));
    max1 = { Lines[0][0],Lines[0][1],Lines[0][2],Lines[0][3] };
    for (size_t i = 0; i < Lines.size(); i++)
    {
        double lineLength = getLineLength(Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]));
        if (lineLength > max) {
            max = lineLength;
            max2 = max1;
            max1 = { Lines[i][0],Lines[i][1],Lines[i][2], Lines[i][3] };
        }
        if (lineLength > 10 && lineLength < 80) {
            line(OutputImage, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]), Scalar(0, 0, 255), 2, 8);
        }
    }
    cout << max1 << " " << max2 << " " << max << endl;
    imshow("OutputImage", OutputImage);
    waitKey(0);
    destroyAllWindows();
}

double getAngle(Point angle, Point edge1, Point edge2) {
    double vec1_x = edge1.x - angle.x;
    double vec1_y = edge1.y - angle.y;
    double vec2_x = edge2.x - angle.x;
    double vec2_y = edge2.y - angle.y;
    double cos = (vec1_x * vec2_x + vec1_y * vec2_y) / (sqrt(pow(vec1_x, 2) + pow(vec1_y, 2)) * sqrt(pow(vec2_x, 2) + pow(vec2_y, 2)));
    double result = acos(cos) / CV_PI * 180;
    return result;
}
void detectFishboneLine(string imgPath) {
    Mat image = imread(imgPath);
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_RGB2GRAY);
    imshow("grayImage", grayImage);
    Mat binaryImage;
    threshold(grayImage, binaryImage, 210, 255, THRESH_BINARY);
    imshow("binaryImage", binaryImage);

    Mat dst0 = Mat::zeros(binaryImage.size(), CV_8UC3);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binaryImage, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Mat contoursImage = image.clone();
    drawContours(contoursImage, contours, -1, Scalar(0, 0, 0), 3);
    imshow("contoursImage", contoursImage);
    vector<vector<Point>> points(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        approxPolyDP(contours[i], points[i], 3, true);
    }

    for (int i = 0; i < contours.size(); i++) {
        if (points[i].size() == 4) {
            Point p1 = points[i][0];
            Point p2 = points[i][1];
            Point p3 = points[i][2];
            Point p4 = points[i][3];
            double sum = 0;
            double a1 = getAngle(p1, p2, p4);
            double a2 = getAngle(p2, p1, p3);
            double a3 = getAngle(p3, p2, p4);
            double a4 = getAngle(p4, p3, p1);
            sum = a1 + a2 + a3 + a4;
            if (sum > 300 && sum < 400) {
                //cout<<"###"<<i<<points[i]<<endl;
                drawContours(image, points, i, Scalar(0, 255, 255), 3);
            }
        }

    }
    imshow("output", image);

    waitKey(0);
    destroyAllWindows();
}
void GetROI(Mat src, Mat& ROI)
{
    int width = src.cols;
    int height = src.rows;

    //获取车道ROI区域，只对该部分进行处理
    vector<Point>pts;
    Point ptA((width / 23) * 2, (height / 15) *  15);
    Point ptB((width / 23) * 2, (height / 15) * 14.8);
    Point ptC((width / 23) * 4, (height / 15) * 10);
    Point ptD((width / 23) * 9, (height / 15) * 9
    
    );
    Point ptE((width / 23) * 17, (height / 15) * 13);
    Point ptF((width / 23) * 17, (height / 15) * 14);
    pts = { ptA ,ptB,ptC,ptD,ptE, ptF };

    //opencv4版本 fillPoly需要使用vector<vector<Point>>
    vector<vector<Point>>ppts;
    ppts.push_back(pts);

    Mat mask = Mat::zeros(src.size(), src.type());
    fillPoly(mask, ppts, Scalar::all(255));

    src.copyTo(ROI, mask);
}

void DetectRoadLine(Mat src, Mat& ROI)
{
    Mat gray;
    cvtColor(ROI, gray, COLOR_BGR2GRAY);

    Mat thresh;
    threshold(gray, thresh, 180, 255, THRESH_BINARY);

    vector<Point>left_line;
    vector<Point>right_line;

    //左车道线
    for (int i = 0; i < thresh.cols / 2; i++)
    {
        for (int j = thresh.rows / 2; j < thresh.rows; j++)
        {
            if (thresh.at<uchar>(j, i) == 255)
            {
                left_line.push_back(Point(i, j));
            }
        }
    }
    //右车道线
    for (int i = thresh.cols / 2; i < thresh.cols; i++)
    {
        for (int j = thresh.rows / 2; j < thresh.rows; j++)
        {
            if (thresh.at<uchar>(j, i) == 255)
            {
                right_line.push_back(Point(i, j));
            }
        }
    }

    //车道绘制
    if (left_line.size() > 0 && right_line.size() > 0)
    {
        Point B_L = (left_line[0]);
        Point T_L = (left_line[left_line.size() - 1]);
        Point T_R = (right_line[0]);
        Point B_R = (right_line[right_line.size() - 1]);

        circle(src, B_L, 10, Scalar(0, 0, 255), -1);
        circle(src, T_L, 10, Scalar(0, 255, 0), -1);
        circle(src, T_R, 10, Scalar(255, 0, 0), -1);
        circle(src, B_R, 10, Scalar(0, 255, 255), -1);

        line(src, Point(B_L), Point(T_L), Scalar(0, 255, 0), 10);
        line(src, Point(T_R), Point(B_R), Scalar(0, 255, 0), 10);

        vector<Point>pts;
        pts = { B_L ,T_L ,T_R ,B_R };
        vector<vector<Point>>ppts;
        ppts.push_back(pts);
        fillPoly(src, ppts, Scalar(133, 230, 238));
    }
}

int main() {

    int model = 1;

    if (model == PictureMode)
    {
        string imgPath = ".\\diamondShapedRoadMarkings.png";
        //string imgPath1 = ".\\straightRoadMarkings.jpg";
        string imgPath1 = ".\\ygx.jpg";
        string imgPathc = ".\\a1.jpg";
        int choice;
        cout << "Detect white dotted line input 1, detect fishbone line input 2:" << endl;
        cin >> choice;
        if (choice == 1)
        {
            detectWhiteDottedLine2(imgPath1);
        }
        else if (choice == 2)
        {
            detectFishboneLine(imgPath1);
        }
    }
    if (model == VideoMode)
    {
        VideoCapture capture;
        capture.open(".\\video.mp4");

        if (!capture.isOpened())
        {
            cout << "Can not open video file!" << endl;
            system("pause");
            return -1;
        }

        Mat frame, image;
        while (capture.read(frame))
        {
            char key = waitKey(10);
            if (key == 27)
            {
                break;
            }
            GetROI(frame, image);

            DetectRoadLine(frame, image);
            resize(frame,frame, Size(), 0.5, 0.5);
            imshow("frame", frame);
        }

        capture.release();
        destroyAllWindows();
        system("pause");
    }
    

    return 0;
}

