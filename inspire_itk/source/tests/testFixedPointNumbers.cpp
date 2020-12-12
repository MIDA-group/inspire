
#include "../common/quantization.h"

#include <vector>
#include <iostream>

namespace TestQuantization
{
    void RunTest()
    {
        std::vector<double> x;
        for(unsigned int i = 1; i <= 1000; ++i)
        {
            x.push_back(1.0/(i));
            x.push_back(-(1.0 + 1e-7)/(i));
        }

        std::vector<double> y;
        for(unsigned int i = 1; i <= 1000; ++i)
        {
            y.push_back(1.0/(i));
        }
        for(unsigned int i = 1; i <= 1000; ++i)
        {
            y.push_back(-(1.0 + 1e-7)/(i));
        }
        
        double xsum = 0.0;
        for(size_t i = 0; i < x.size(); ++i)
        {
            xsum += x[i];
        }

        double ysum = 0.0;
        for(size_t i = 0; i < y.size(); ++i)
        {
            ysum += y[i];
        }

        double xavg = (xsum/x.size());
        double yavg = (ysum/y.size());
        std::cout << xavg << std::endl;
        std::cout << yavg << std::endl;

        FixedPointNumber xfpsum = 0;
        FixedPointNumber yfpsum = 0;
        for(size_t i = 0; i < x.size(); ++i)
        {
            xfpsum = xfpsum + FixedPointFromDouble(x[i]);
            yfpsum = yfpsum + FixedPointFromDouble(y[i]);
        }

        double xfpavg = DoubleFromFixedPoint(xfpsum)/x.size();
        double yfpavg = DoubleFromFixedPoint(yfpsum)/y.size();
        std::cout << xfpavg << std::endl;
        std::cout << yfpavg << std::endl;

        std::cout << (xavg==yavg) << std::endl;
        std::cout << (xfpavg==yfpavg) << std::endl;
    }
}
