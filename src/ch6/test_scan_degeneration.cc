//
// Created by yixfeng on 6/07/23.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ch6/degeneration_detection.h"
#include "ch6/lidar_2d_utils.h"
#include "common/io_utils.h"

DEFINE_string(bag_path, "./dataset/sad/2dmapping/floor1.bag", "数据包路径");
DEFINE_int32(core_minpts, 10, "核心点邻域内最少点数");
DEFINE_double(eps, 0.2, "邻域大小");
DEFINE_int32(minpts_per_cluster, 40, "每个cluster最少点数");
DEFINE_int32(maxpts_per_cluster, 10000, "每个cluster最多点数");

// 测试从rosbag中读取2d scan并进行退化检测的结果
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path);
    Scan2d::Ptr last_scan = nullptr, current_scan = nullptr;

    /// 我们将上一个scan与当前scan进行配准
    rosbag_io
        .AddScan2DHandle("/pavo_scan_bottom",
                         [&](Scan2d::Ptr scan) {
                             sad::DegenDetection degen_det;
                             degen_det.SetScan(scan);
                             degen_det.BuildKdTree();
                             degen_det.InitDBSCAN(FLAGS_core_minpts, FLAGS_minpts_per_cluster,
                                                  FLAGS_maxpts_per_cluster, FLAGS_eps);

                             cv::Mat image;
                             degen_det.ShowClustering(image, 1000, 40.0);
                             cv::Mat image_scan;
                             sad::Visualize2DScan(scan, SE2(), image_scan, Vec3b(255, 0, 0));
                             cv::imshow("clustering", image);
                             cv::imshow("scan", image_scan);

                             if (degen_det.DetectDegeneration()) {
                                 LOG(INFO) << "检测到退化场景";
                                 cv::imwrite(
                                     "./data/ch6/degen_" + std::to_string(degen_det.GetDegenCount()) + ".png", image);
                                 degen_det.AddCount();
                             }
                             cv::waitKey(20);
                             return true;
                         })
        .Go();

    return 0;
}