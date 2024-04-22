#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main() {
    setlocale(LC_ALL, "Russian");

    Mat image = imread("C:/Users/Катя/Desktop/Python/test (1).jpg");

    resize(image, image, Size(), 0.2, 0.2);

    vector<Mat> cardsImages;
    vector<string> cardsNames;
    vector<Mat> cardsDescriptors;

    string cardsFolder = "C:/Users/Катя/source/repos/Project1/Project1/Cards/";
    for (int i = 1; i <= 7; ++i) {
        string filename = cardsFolder + "card" + to_string(i) + ".jpg";
        Mat cardImage = imread(filename);
        if (cardImage.empty()) {
            cerr << "Failed to load image: " << filename << endl;
            continue;
        }
        cardsImages.push_back(cardImage);
    }

    cardsNames.push_back("9_kresti");
    cardsNames.push_back("korol_bubi");
    cardsNames.push_back("tuz_chervi");
    cardsNames.push_back("dama_picki");
    cardsNames.push_back("valet_chervi");
    cardsNames.push_back("6_picki");
    cardsNames.push_back("tuz_bubi");

    Ptr<ORB> detector = ORB::create();

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, gray, Size(5, 5), 0);

    Mat descriptors;
    vector<KeyPoint> keypoints;
    detector->detectAndCompute(gray, noArray(), keypoints, descriptors);
    Mat edges;
    Canny(gray, edges, 200, 90);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    drawContours(image, contours, -1, Scalar(0, 255, 0), 1);

    for (size_t i = 0; i < cardsNames.size(); i++) {
        Mat cardImage = cardsImages[i];
        Mat cardGray;
        cvtColor(cardImage, cardGray, COLOR_BGR2GRAY);

        Mat cardDescriptors;
        vector<KeyPoint> cardKeypoints;
        detector->detectAndCompute(cardGray, noArray(), cardKeypoints, cardDescriptors);

        Mat cardEdges;
        Canny(cardGray, cardEdges, 30, 90);

        vector<vector<Point>> cardContours;
        vector<Vec4i> cardHierarchy;
        findContours(cardEdges, cardContours, cardHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        drawContours(cardImage, cardContours, -1, Scalar(0, 0, 255), 1);

        putText(image, cardsNames[i], keypoints[i].pt, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
    }

    imshow("result", image);
    waitKey(0);

    return 0;
}
