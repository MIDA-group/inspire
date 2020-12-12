
#include "testPointSamplers.cpp"
//#include "testValueSamplers.cpp"
#include "testRegistration.cpp"
#include "testFixedPointNumbers.cpp"

int main(int argc, char** argv) {
    //PointSamplerTests::RunPointSamplersTests();
    //ValueSamplerTests::RunValueSamplersTests();
    TestRegistration::RunTest(argc, argv);
    //TestQuantization::RunTest();
    
    return 0;
}