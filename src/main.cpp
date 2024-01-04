#include <iostream>
#include <stdio.h>
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#define LEFTCAMINDEX 0
#define RIGHTCAMINDEX 1
#define BOARDWIDTH 11
#define BOARDHEIGHT 8
#define CHECKERSIZE 17 // [mm]
#define SQUARESIZE (CHECKERSIZE * CHECKERSIZE) // [sqmm]
#define NUMBEROFIMAGES 20

class Calibrator
{
public:
    void calibrate();
    Calibrator(const unsigned int leftCameraIndex, const unsigned int rightCameraIndex);
    ~Calibrator();
private:
    void writeCalibrationData();
    void captureImagesForCalibration();
    std::pair <cv::Mat, cv::Mat> getFramesFromCaptures();
    void viewCombinedCameraFeeds();
    void startVideoCaptures(const unsigned int leftCameraIndex, const unsigned int rightCameraIndex);
    void drawAxis(cv::Mat& matImg, cv::InputArray K, cv::InputArray D, cv::InputArray rvec, cv::InputArray tvec, float checkersize);

    cv::Size boardSize{BOARDWIDTH - 1, BOARDHEIGHT - 1};
    cv::VideoCapture capLeft;
    cv::VideoCapture capRight;
    const unsigned int leftCameraIndex, rightCameraIndex;
    std::vector<std::vector<cv::Point2f>> imagePointsL, imagePointsR;
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Mat matImgL, matImgR;
    cv::Mat KL, DL;
    cv::Mat KR, DR;
    std::vector<cv::Mat> rvecsL, tvecsL;
    std::vector<cv::Mat> rvecsR, tvecsR;
    int flag = cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT + cv::CALIB_FIX_ASPECT_RATIO;
};

Calibrator::Calibrator(const unsigned int leftCameraIndex, const unsigned int rightCameraIndex)
    : leftCameraIndex(leftCameraIndex), rightCameraIndex(rightCameraIndex)
{

}

Calibrator::~Calibrator()
{
    capLeft.release();
    capRight.release();
}

void getCameraIndexes()
{
    for (int cameraIndex = 0; cameraIndex < 10; ++cameraIndex) {
        cv::VideoCapture cap(cameraIndex);

        // Check if the camera opened successfully
        if (cap.isOpened()) {
            std::cout << "Camera found at index: " << cameraIndex << std::endl;

            // Release the VideoCapture
            cap.release();
        }
    }
}

std::pair <cv::Mat, cv::Mat> Calibrator::getFramesFromCaptures()
{
    // Read a frame from the camera
    cv::Mat frameLeft;
    cv::Mat frameRight;
    capLeft >> frameLeft;
    capRight >> frameRight;     
    // Check if the frame is empty
    if (frameLeft.empty())
    {
        std::cerr << "Error reading left frame from camera." << std::endl;
    }
    if (frameRight.empty())
    {
        std::cerr << "Error reading right frame from camera." << std::endl;
    }
    return std::make_pair(frameLeft, frameRight);
}

void Calibrator::viewCombinedCameraFeeds()
{
    // Create a window to display the camera feed
    cv::namedWindow("Dual Webcams", cv::WINDOW_FREERATIO);
    while (true)
    {
        auto [frameLeft, frameRight] = getFramesFromCaptures();
        // Flip frames vertically
        cv::flip(frameLeft, frameLeft, -1);
        cv::flip(frameRight, frameRight, -1);
        // Combine frames into one
        cv::Mat combinedFrame;
        cv::hconcat(frameLeft, frameRight, combinedFrame);
        // Display the combined frame
        cv::imshow("Dual Webcams", combinedFrame);
        // To exit press esc
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
    cv::destroyAllWindows();
}

void Calibrator::startVideoCaptures(const unsigned int leftCameraIndex, const unsigned int rightCameraIndex)
{
    cv::VideoCapture tempCapLeft(leftCameraIndex);
    cv::VideoCapture tempCapRight(rightCameraIndex);
    if (not tempCapLeft.isOpened())
    {
        std::cerr << "Error opening the left camera." << std::endl;
    }
    if (not tempCapRight.isOpened())
    {
        std::cerr << "Error opening the right camera." << std::endl;
    }
    this->capLeft = tempCapLeft;
    this->capRight = tempCapRight;
}

void Calibrator::drawAxis(cv::Mat& matImg, cv::InputArray K, cv::InputArray D, cv::InputArray rvec, cv::InputArray tvec, float checkersize)
{
    // project axis points
    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f(0, 0, 0));
    axisPoints.push_back(cv::Point3f(checkersize, 0, 0));
    axisPoints.push_back(cv::Point3f(0, checkersize, 0));
    axisPoints.push_back(cv::Point3f(0, 0, checkersize));
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, rvec, tvec, K, K, imagePoints);

    // draw axis lines
    cv::line(matImg, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
    cv::line(matImg, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
    cv::line(matImg, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
}

void Calibrator::writeCalibrationData()
{
    //Writing the data in a YAML file
    std::string outputFile = "calibration.txt";
    cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
    if(not fs.isOpened())
    {
        std::cerr << "Error saving calibration data." << std::endl;
        return;
    }
    fs << "KL" << KL;
    fs << "DL" << DL;
    fs << "KR" << KR;
    fs << "DR" << DR;
    fs << "board_width" << BOARDWIDTH;
    fs << "board_height" << BOARDHEIGHT;
    fs << "square_size" << CHECKERSIZE;
    printf("Done writing calibration data\n");
}

void Calibrator::captureImagesForCalibration()
{
    // Loop through video stream until we capture enough images (num_imgs) of checkerboard
    while(true)
    {
        capLeft >> matImgL;
        capRight >> matImgR;
        if(matImgL.empty() or matImgR.empty())
        {
            break;
        }
        // Сheckerboard corner coordinates in the image
        std::vector<cv::Point2f> cornersL;
        std::vector<cv::Point2f> cornersR;
        // Here we find all the corner points of each image and their corresponding 3D world points
        // and prepare the corresponding vectors.
        // Find all the checkerboard corners
        bool foundL = cv::findChessboardCorners(matImgL, boardSize, cornersL, cv::CALIB_CB_ADAPTIVE_THRESH);
        bool foundR = cv::findChessboardCorners(matImgR, boardSize, cornersR, cv::CALIB_CB_ADAPTIVE_THRESH);
        cv::Size winSize = cv::Size( 12, 12 );
        cv::Size zeroZone = cv::Size( -1, -1 );
        cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1 );
        if(foundL and foundR)
        {
            // Convert image to grayscale images
            cv::Mat grayL;
            cvtColor(matImgL, grayL, cv::COLOR_BGR2GRAY);
            // Find more exact corner positions (more exact than integer pixels)
            cv::cornerSubPix( grayL, cornersL, winSize, zeroZone, criteria );
            // This function helps to visualize the checkerboard corners found (optional)
            cv::drawChessboardCorners(matImgL, boardSize, cv::Mat(cornersL), foundL);

            // Convert image to grayscale images
            cv::Mat grayR;
            cvtColor(matImgR, grayR, cv::COLOR_BGR2GRAY);
            // Find more exact corner positions (more exact than integer pixels)
            cv::cornerSubPix( grayR, cornersR, winSize, zeroZone, criteria );
            // This function helps to visualize the checkerboard corners found (optional)
            cv::drawChessboardCorners(matImgR, boardSize, cv::Mat(cornersR), foundR);

            // Prepare the objectPoints and imagePoints vectors
            std::vector<cv::Point3f> obj;
            for (int i = 0; i < BOARDHEIGHT - 1; i++)
            {
                for (int j = 0; j < BOARDWIDTH - 1; j++)
                {
                    obj.push_back(cv::Point3f((float)j * CHECKERSIZE, (float)i * CHECKERSIZE, 0));
                }
            }
            std::cout <<"Found corners!" << std::endl;
            imagePointsL.push_back(cornersL);
            imagePointsR.push_back(cornersR);
            objectPoints.push_back(obj);
        }
        cv::imshow("Camera left", matImgL);
        cv::imshow("Camera right", matImgR);
        cv::waitKey(50);
        if (imagePointsL.size() == NUMBEROFIMAGES and imagePointsR.size() == NUMBEROFIMAGES)
        {
            break;
        }
    }
}

void Calibrator::calibrate()
{
    captureImagesForCalibration();
    printf("Starting Calibration\n");
    int flag = cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT + cv::CALIB_FIX_ASPECT_RATIO;
    double reprojectionErrorL = cv::calibrateCamera(objectPoints, imagePointsL, matImgL.size(), KL, DL, rvecsL, tvecsL, flag);
    double reprojectionErrorR = cv::calibrateCamera(objectPoints, imagePointsR, matImgR.size(), KR, DR, rvecsR, tvecsR, flag);
    std::cout << "reprojectionErrorL: " << reprojectionErrorL << std::endl;
    std::cout << "reprojectionErrorR: " << reprojectionErrorR << std::endl;
    while(true)
    {
        capLeft >> matImgL;
        capRight >> matImgR;
        if(matImgL.empty() or matImgR.empty())
        {
            std::cerr << "Error reading from camera capture after calibration." << std::endl;
            break;
        }
        //Found chessboard corners
        std::vector<cv::Point2f> imagePointsL, imagePointsR;
        bool foundL = cv::findChessboardCorners(matImgL, boardSize, imagePointsL, cv::CALIB_CB_ADAPTIVE_THRESH);
        bool foundR = cv::findChessboardCorners(matImgR, boardSize, imagePointsR, cv::CALIB_CB_ADAPTIVE_THRESH);
        if(foundL or foundR)
        {
            cv::drawChessboardCorners(matImgL, boardSize, cv::Mat(imagePointsL), foundL);
            cv::drawChessboardCorners(matImgR, boardSize, cv::Mat(imagePointsR), foundR);

            std::vector<cv::Point3f> obj;
            for (int i = 0; i < BOARDHEIGHT - 1; i++)
            {
                for (int j = 0; j < BOARDWIDTH - 1; j++)
                {
                    obj.push_back(cv::Point3f((float)j * CHECKERSIZE, (float)i * CHECKERSIZE, 0));
                }
            }
            //SolvePnP
            cv::Mat rvecL, tvecL;
            cv::Mat rvecR, tvecR;
            cv::solvePnP(obj, imagePointsL, KL, DL, rvecL, tvecL);
            cv::solvePnP(obj, imagePointsR, KR, DR, rvecR, tvecR);

            drawAxis(matImgL, KL, DL, rvecL, tvecL, CHECKERSIZE);
            drawAxis(matImgR, KR, DR, rvecR, tvecR, CHECKERSIZE);
            std::cout << "Distance to chessboard from left camera: "  << cv::norm(tvecL) << std::endl;
            std::cout << "Distance to chessboard from right camera: "  << cv::norm(tvecR) << std::endl;
        }

        cv::imshow("Camera left", matImgL);
        cv::imshow("Camera right", matImgR);
        auto c = cv::waitKey(30);

        // 27 == ESC
        // !! The window with the image displayed in it must have focus (i.e. be selected) when you press the key
        if(c == 27) {
            break;
        }
    }
    writeCalibrationData();
}

int main()
{
    Calibrator calib(LEFTCAMINDEX, RIGHTCAMINDEX);
    calib.calibrate();
    return 0;
}
