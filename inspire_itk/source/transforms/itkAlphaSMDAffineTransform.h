
#ifndef ITK_ALPHA_SMD_AFFINE_TRANSFORM_H
#define ITK_ALPHA_SMD_AFFINE_TRANSFORM_H

// Core ITK facilities
#include "itkPoint.h"
#include "itkMath.h"
#include "itkImage.h"
#include "itkSmartPointer.h"

#include "itkAffineTransform.h"
#include "itkRigid2DTransform.h"

//DV/DT = DV/DS * DS/DT

namespace itk {
    template <typename ValueType>
    class AlphaSMDRigid2DTransform {
        public:
        static const unsigned int NParams = 3U; // Number of parameters
        static const unsigned int Dim = 2U;

        typedef itk::Point<ValueType, 2U> Point;
        typedef itk::Point<ValueType, 2U+1> HPoint;
        typedef itk::FixedArray<ValueType, 3U> ParamArrayType;
        typedef itk::Matrix<ValueType, Dim, Dim> Matrix;
        typedef itk::Matrix<ValueType, Dim+1, Dim+1> HMatrix;

        void Begin() {
            const double theta = m_T[0];
            
            m_ForwardMatrix(0, 0) = cos(theta);
            m_ForwardMatrix(1, 0) = -sin(theta);
            m_ForwardMatrix(0, 1) = -m_ForwardMatrix(1, 0);
            m_ForwardMatrix(1, 1) = m_ForwardMatrix(0, 0);
            m_ForwardTranslation[0] = m_T[1];
            m_ForwardTranslation[1] = m_T[2];

            m_InverseMatrix(0, 0) = cos(-theta);
            m_InverseMatrix(1, 0) = -sin(-theta);
            m_InverseMatrix(0, 1) = -m_InverseMatrix(1, 0);
            m_InverseMatrix(1, 1) = m_InverseMatrix(0, 0);
            m_InverseTranslation[0] = -m_T[1];
            m_InverseTranslation[1] = -m_T[2];

            m_ForwardDiffMatrix(0, 0) = -sin(theta);
            m_ForwardDiffMatrix(1, 0) = -cos(theta);
            m_ForwardDiffMatrix(0, 1) = -m_ForwardDiffMatrix(1, 0);
            m_ForwardDiffMatrix(1, 1) = m_ForwardDiffMatrix(0, 0);

            m_InverseDiffMatrix(0, 0) = sin(-theta);
            m_InverseDiffMatrix(1, 0) = cos(-theta);
            m_InverseDiffMatrix(0, 1) = -m_InverseDiffMatrix(1, 0);
            m_InverseDiffMatrix(1, 1) = m_InverseDiffMatrix(0, 0);
        }

        inline Point TransformForward(Point p) {
            return m_ForwardMatrix * p + m_ForwardTranslation;
        }
        inline Point TransformInverse(Point p) {
            return m_InverseMatrix * p + m_InverseTranslation;
        }

        inline void DiffForward(Point p, ParamArrayType* out) {
            Point theta_p = m_ForwardDiffMatrix * p;
            out[0][0] = theta_p[0];
            out[1][0] = theta_p[1];

            out[0][1] = 1.0;
            out[1][1] = 0.0;

            out[0][2] = 0.0;
            out[1][2] = 1.0;
        }
        
        inline void DiffInverse(Point p, ParamArrayType* out) {
            Point q = m_InverseDiffMatrix * (p + m_InverseTranslation);

            Point x_base;
            x_base[0] = -1.0;
            x_base[1] = 0.0;
            x_base = m_InverseMatrix * x_base;

            Point y_base;
            y_base[0] = 0.0;
            y_base[1] = -1.0;
            y_base = m_InverseMatrix * y_base;

            out[0][0] = q[0];
            out[1][0] = q[1];

            out[0][1] = x_base[0];
            out[1][1] = x_base[1];

            out[0][2] = y_base[0];
            out[1][2] = y_base[1];           
        }

       //DV/DT = DV/DS * DS/DT
        template <typename OutArrayType>
        inline void TotalDiffForward(Point p, itk::CovariantVector<ValueType, Dim> spatialGrad, OutArrayType& out, ValueType weight, bool addDerivative = true) {
            ParamArrayType d[2U];

            DiffForward(p, d);

            for(unsigned int i = 0; i < NParams; ++i) {
                ValueType acc = 0;

                for(unsigned int j = 0; j < Dim; ++j) {
                    acc += weight * spatialGrad[j] * d[j][i];
                }

                if(addDerivative)
                    out[i] += acc;
                else
                    out[i] = acc;
            }
        }
        
        template <typename OutArrayType>
        inline void TotalDiffInverse(Point p, itk::CovariantVector<ValueType, 2U> spatialGrad, OutArrayType& out, ValueType weight, bool addDerivative = true) {
            ParamArrayType d[2U];

            DiffInverse(p, d);

            for(unsigned int i = 0; i < NParams; ++i) {
                ValueType acc = 0;

                for(unsigned int j = 0; j < Dim; ++j) {
                    acc += weight * spatialGrad[j] * d[j][i];
                }

                if(addDerivative)
                    out[i] += acc;
                else
                    out[i] = acc;
            }
        }      

        inline static unsigned int GetParamCount() { return NParams; }

        void SetIdentity() {
            this->m_T.Fill(0);
        }

        //inline void DiffInverse(Point p, ParamArrayType* out) { assert(false); }
        // and TotalDiffInverse, similar to TotalDiffForward        

        ValueType GetParam(unsigned int index) {
            return this->m_T[index];
        }

        void SetParam(unsigned int index, ValueType x) {
            this->m_T[index] = x;
        }

        typename itk::Rigid2DTransform<ValueType>::Pointer ConvertToITKTransform() const {
            typedef itk::Rigid2DTransform<ValueType> ITKTransformType;
            typedef typename ITKTransformType::Pointer ITKTransformPointer;

            ITKTransformPointer transform = ITKTransformType::New();

            itk::OptimizerParameters<double> param(3U);

            param[0] = -m_T[0];
            param[1] = m_T[1];
            param[2] = m_T[2];

            transform->SetParameters(param);

            return transform;
        }

        void ConvertFromITKTransform(
            typename itk::Transform<double, 2U, 2U>::Pointer transform)
        {
            if(transform) {
                assert(transform->GetNumberOfParameters() == NParams);

                //TODO: Add verification that transform is a Rigid2DTransform
                SetParam(0, -transform->GetParameters()[0]);
                for(unsigned int i = 1; i < 3; ++i)
                    SetParam(i, transform->GetParameters()[i]);
            }
        }

        protected:
        ParamArrayType m_T;

        Matrix m_ForwardMatrix;
        Matrix m_InverseMatrix;
        Matrix m_ForwardDiffMatrix;
        Matrix m_InverseDiffMatrix;

		itk::Vector<double, 2U> m_ForwardTranslation;
		itk::Vector<double, 2U> m_InverseTranslation;        
    };


    template <typename ValueType, unsigned int Dim>
    class AlphaSMDAffineTransformBase {
        public:
        static const unsigned int NParams = Dim * (Dim + 1); // Number of parameters

        typedef itk::Point<ValueType, Dim> Point;
        typedef itk::Point<ValueType, Dim+1> HPoint;
        typedef itk::FixedArray<ValueType, Dim*(Dim+1)> ParamArrayType;
        typedef itk::Matrix<ValueType, Dim, Dim> Matrix;
        typedef itk::Matrix<ValueType, Dim+1, Dim+1> HMatrix;

        void BeginBase() {
            HMatrix hForwardMatrix;
            HMatrix hInverseMatrix;

            hForwardMatrix.Fill(0);
            
            unsigned int pind = 0;
            for(unsigned int i = 0; i < Dim; ++i) {
                for(unsigned int j = 0; j < Dim; ++j) {
                    hForwardMatrix(i, j) = this->m_T[pind++];
                }
                hForwardMatrix(i, Dim) = this->m_T[pind++];
            }

            hForwardMatrix(Dim, Dim) = 1;

            // TODO: Make sure Handling exception works!

            try {
                hInverseMatrix = hForwardMatrix.GetInverse();
            } catch(itk::ExceptionObject e) {
                hInverseMatrix.Fill(0);
            }

            for(unsigned int i = 0; i < Dim; ++i) {
                for(unsigned int j = 0; j < Dim; ++j) {
                    m_ForwardMatrix(i, j) = hForwardMatrix(i, j);
                    m_InverseMatrix(i, j) = hInverseMatrix(i, j);
                }
				m_ForwardTranslation[i] = hForwardMatrix(i, Dim);
				m_InverseTranslation[i] = hInverseMatrix(i, Dim);
            }
        }

        //void Begin() { assert(false); }

        inline Point TransformForward(Point p) {
            return m_ForwardMatrix * p + m_ForwardTranslation;
        }
        inline Point TransformInverse(Point p) {
            return m_InverseMatrix * p + m_InverseTranslation;
        }

        inline void DiffForward(Point p, ParamArrayType* out) {
            for(unsigned int i = 0; i < Dim; ++i) {
                out[i].Fill(0);
                unsigned int start = i * (Dim + 1);
                for(unsigned int j = 0; j < Dim; ++j) {
                    out[i][start + j] = p[j];
                }
                out[i][start + Dim] = 1;
            }
        }

        //DV/DT = DV/DS * DS/DT
        template <typename OutArrayType>
        inline void TotalDiffForward(Point p, itk::CovariantVector<ValueType, Dim> spatialGrad, OutArrayType& out, ValueType weight, bool addDerivative = true) {
            ParamArrayType d[Dim];

            DiffForward(p, d);

            for(unsigned int i = 0; i < NParams; ++i) {
                ValueType acc = 0;

                for(unsigned int j = 0; j < Dim; ++j) {
                    acc += weight * spatialGrad[j] * d[j][i];
                }

                if(addDerivative)
                    out[i] += acc;
                else
                    out[i] = acc;
            }
        }
        
        // Implement in derived classes (specializations):
        //inline void DiffInverse(Point p, ParamArrayType* out) { assert(false); }
        // and TotalDiffInverse, similar to TotalDiffForward

        inline static unsigned int GetParamCount() { return Dim*(Dim+1); }

        void SetIdentity() {
            this->m_T.Fill(0);
            unsigned int it = 0;
            for(unsigned int i = 0; i < Dim; ++i) {
                this->m_T[it] = 1;
                it += (Dim+2);
            }
        }

        ValueType GetParam(unsigned int index) {
            return this->m_T[index];
        }

        void SetParam(unsigned int index, ValueType x) {
            this->m_T[index] = x;
        }

        typename itk::AffineTransform<ValueType, Dim>::Pointer ConvertToITKTransform() const {
            typedef itk::AffineTransform<ValueType, Dim> ITKTransformType;
            typedef typename ITKTransformType::Pointer ITKTransformPointer;

            ITKTransformPointer transform = ITKTransformType::New();

            itk::OptimizerParameters<double> param(NParams);

            // Matrix

            unsigned int km = 0;
            for(unsigned int i = 0; i < Dim; ++i) {
                unsigned int start = i * (Dim + 1);
                for(unsigned int j = 0; j < Dim; ++j) {
                    param[km++] = this->m_T[start + j];
                }
            }

            // Translation
            
            unsigned int kt = Dim*Dim;
            for(unsigned int i = 0; i < Dim; ++i) {
                param[kt++] = this->m_T[Dim + (Dim+1) * i];
            }
            //std::cout << param << std::endl;
            transform->SetParameters(param);

            return transform;
        }

        void ConvertFromITKTransform(
            typename itk::Transform<double, Dim, Dim>::Pointer transform)
        {
            if(transform) {
                if(Dim == 2U && transform->GetNumberOfParameters() == 3U) { // Rigid
                //TODO: Verify transform class name
                    double theta = transform->GetParameters()[0];
                    double tx = transform->GetParameters()[1];
                    double ty = transform->GetParameters()[2];

                    this->m_T[0] = cos(theta);
                    this->m_T[1] = -sin(theta);
                    this->m_T[2] = tx;
                    this->m_T[3] = sin(theta);
                    this->m_T[4] = cos(theta);
                    this->m_T[5] = ty;

                    return;
                }

                assert(transform->GetNumberOfParameters() == NParams);

                // Matrix

                unsigned int km = 0;
                for(unsigned int i = 0; i < Dim; ++i) {
                    unsigned int start = i * (Dim + 1);
                    for(unsigned int j = 0; j < Dim; ++j) {
                        this->m_T[start + j] = transform->GetParameters()[km++];
                    }
                }

                // Translation
            
                unsigned int kt = Dim*Dim;
                for(unsigned int i = 0; i < Dim; ++i) {
                    this->m_T[Dim + (Dim+1) * i] = transform->GetParameters()[kt++];
                }
            }
        }

        protected:
        ParamArrayType m_T;

        Matrix m_ForwardMatrix;
        Matrix m_InverseMatrix;

		itk::Vector<double, Dim> m_ForwardTranslation;
		itk::Vector<double, Dim> m_InverseTranslation;
    };

    template <typename ValueType, unsigned int Dim>
    class AlphaSMDAffineTransform : public AlphaSMDAffineTransformBase<ValueType, Dim> { };

    // 2D-specialization

    template <typename ValueType>
    class AlphaSMDAffineTransform<ValueType, 2U> : public AlphaSMDAffineTransformBase<ValueType, 2U> {
        public:
			static const unsigned int Dim = 2U;
            static const unsigned int NParams = Dim * (Dim + 1);
        typedef itk::Point<ValueType, Dim> Point;
        typedef itk::FixedArray<ValueType, Dim*(Dim+1)> ParamArrayType;
        
/*	        double a00 = t[0];
	        double a10 = this->m_T[1];
	        double tx = this->m_T[2];
	        double a01 = this->m_T[3];
	        double a11 = this->m_T[4];
	        double ty = this->m_T[5];**/

	        // format: a in out
        
        void Print() {
            for(unsigned int i = 0; i < AlphaSMDAffineTransformBase<ValueType, 2U>::GetParamCount(); ++i)
                std::cout << "Parameter " << i << " is " << this->m_T[i] << std::endl;
        }

        void Begin() {
            this->BeginBase();

            double dtm = this->m_T[0] * this->m_T[4] - this->m_T[3] * this->m_T[1];
            double dtmInv;
            if (dtm != 0.0 && dtm != -0.0)
                dtmInv = 1.0 / dtm;
            else
                dtmInv = 0.0;

            m_Determinant = dtm;
            m_DeterminantRec = dtmInv;
        }

/*	        double a00 = t[0];
	        double a10 = t[1];
	        double tx = t[2];
	        double a01 = t[3];
	        double a11 = t[4];
	        double ty = t[5];**/

	        // format: a in out
        void DiffInverse(Point p, ParamArrayType *out)
        {
            double dtm = m_Determinant;

            double dtmInvSq = m_DeterminantRec * m_DeterminantRec;

            double dx_tx = -this->m_T[4] * m_DeterminantRec;
            double dy_tx = this->m_T[3] * m_DeterminantRec;

            double dx_ty = this->m_T[1] * m_DeterminantRec;
            double dy_ty = -this->m_T[0] * m_DeterminantRec;

            out[0][0] = (-this->m_T[4] * this->m_T[4] * p[0] + this->m_T[1] * this->m_T[4] * p[1] - (this->m_T[1] * this->m_T[5] - this->m_T[2] * this->m_T[4]) * this->m_T[4]) * dtmInvSq;
            out[1][0] = (this->m_T[3] * this->m_T[4] * p[0] + (dtm - this->m_T[0] * this->m_T[4]) * p[1] - this->m_T[5] * dtm - (this->m_T[2] * this->m_T[3] - this->m_T[0] * this->m_T[5]) * this->m_T[4]) * dtmInvSq;

            out[1][1] = (-this->m_T[3] * this->m_T[3] * p[0] + this->m_T[0] * this->m_T[3] * p[1] + (this->m_T[2] * this->m_T[3] - this->m_T[0] * this->m_T[5]) * this->m_T[3]) * dtmInvSq;
            out[0][1] = (this->m_T[3] * this->m_T[4] * p[0] - (dtm + this->m_T[1] * this->m_T[3]) * p[1] + dtm * this->m_T[5] + (this->m_T[1] * this->m_T[5] - this->m_T[2] * this->m_T[4]) * this->m_T[3]) * dtmInvSq;

            out[0][2] = dx_tx;
            out[1][2] = dy_tx;

            out[0][3] = (this->m_T[4] * this->m_T[1] * p[0] - this->m_T[1] * this->m_T[1] * p[1] + (this->m_T[1] * this->m_T[5] - this->m_T[2] * this->m_T[4]) * this->m_T[1]) * dtmInvSq;
            out[1][3] = ((-dtm - this->m_T[3] * this->m_T[1]) * p[0] + this->m_T[0] * this->m_T[1] * p[1] + this->m_T[2] * dtm + (this->m_T[2] * this->m_T[3] - this->m_T[0] * this->m_T[5]) * this->m_T[1]) * dtmInvSq;

            out[0][4] = ((dtm - this->m_T[4] * this->m_T[0]) * p[0] + this->m_T[0] * this->m_T[1] * p[1] - dtm * this->m_T[2] - (this->m_T[1] * this->m_T[5] - this->m_T[2] * this->m_T[4]) * this->m_T[0]) * dtmInvSq;
            out[1][4] = (this->m_T[0] * this->m_T[3] * p[0] - this->m_T[0] * this->m_T[0] * p[1] - (this->m_T[2] * this->m_T[3] - this->m_T[0] * this->m_T[5]) * this->m_T[0]) * dtmInvSq;

            out[0][5] = dx_ty;
            out[1][5] = dy_ty;
        }

        //DV/DT = DV/DS * DS/DT
        template <typename OutArrayType>
        inline void TotalDiffInverse(Point p, itk::CovariantVector<ValueType, Dim> spatialGrad, OutArrayType& out, ValueType weight, bool addDerivative = true) {
            ParamArrayType d[Dim];

            DiffInverse(p, d);

            for(unsigned int i = 0; i < NParams; ++i) {
                ValueType acc = 0;

                for(unsigned int j = 0; j < Dim; ++j) {
                    acc += weight * spatialGrad[j] * d[j][i];
                }

                if(addDerivative)
                    out[i] += acc;
                else
                    out[i] = acc;
            }
        }
           
        protected:
            ValueType m_Determinant;
            ValueType m_DeterminantRec;
    };

    // 3D-specialization
    template <typename ValueType>
    class AlphaSMDAffineTransform<ValueType, 3U> : public AlphaSMDAffineTransformBase<ValueType, 3U>
    {
      public:
        static const unsigned int Dim = 3U;
        static const unsigned int NParams = Dim * (Dim + 1);
        
        typedef itk::Point<ValueType, Dim> Point;
        typedef itk::FixedArray<ValueType, Dim *(Dim + 1)> ParamArrayType;

        void Begin()
        {
            this->BeginBase();

            ValueType a1 = this->m_T[0];
            ValueType a2 = this->m_T[1];
            ValueType a3 = this->m_T[2];
            //ValueType a4 = this->m_T[3];
            ValueType b1 = this->m_T[4];
            ValueType b2 = this->m_T[5];
            ValueType b3 = this->m_T[6];
            //ValueType b4 = this->m_T[7];
            ValueType c1 = this->m_T[8];
            ValueType c2 = this->m_T[9];
            ValueType c3 = this->m_T[10];
            //ValueType c4 = this->m_T[11];

            double dtm = (a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1);
            double dtmInv;
            if (dtm != 0.0 && dtm != -0.0)
                dtmInv = 1.0 / dtm;
            else
                dtmInv = 0.0;

            m_DeterminantRec = dtmInv;
        }

        void DiffInverse(Point p, ParamArrayType *out)
        {
            ValueType x = p[0];
            ValueType y = p[1];
            ValueType z = p[2];

            ValueType a1 = this->m_T[0];
            ValueType a2 = this->m_T[1];
            ValueType a3 = this->m_T[2];
            ValueType a4 = this->m_T[3];
            ValueType b1 = this->m_T[4];
            ValueType b2 = this->m_T[5];
            ValueType b3 = this->m_T[6];
            ValueType b4 = this->m_T[7];
            ValueType c1 = this->m_T[8];
            ValueType c2 = this->m_T[9];
            ValueType c3 = this->m_T[10];
            ValueType c4 = this->m_T[11];
/*
            std::cout << "[" << a1 << " " << a2 << " " << a3 << " " << a4 << "]" << std::endl;
            std::cout << "[" << b1 << " " << b2 << " " << b3 << " " << b4 << "]" << std::endl;
            std::cout << "[" << c1 << " " << c2 << " " << c3 << " " << c4 << "]" << std::endl;
            std::cout << "sqr(2.0) = " << itk::Math::sqr(2.0) << std::endl;
			std::cout << "determ_rec = " << m_DeterminantRec << std::endl;
            */
            ValueType determ_rec = m_DeterminantRec;
            ValueType determ_rec_sq = determ_rec * determ_rec;

            /////////////////////////////////////////////////////////////////////////////////

            ValueType dx_da1=(((b2*c3 - b3*c2)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (x*itk::Math::sqr(b2*c3 - b3*c2))*determ_rec_sq - (z*(a2*b3 - a3*b2)*(b2*c3 - b3*c2))*determ_rec_sq + (y*(a2*c3 - a3*c2)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_da1=((b3*c4 - b4*c3)*determ_rec - ((b2*c3 - b3*c2)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (b3*z)*determ_rec + (c3*y)*determ_rec + (z*(a1*b3 - a3*b1)*(b2*c3 - b3*c2))*determ_rec_sq - (y*(a1*c3 - a3*c1)*(b2*c3 - b3*c2))*determ_rec_sq + (x*(b1*c3 - b3*c1)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dz_da1=-((b2*c4 - b4*c2)*determ_rec - ((b2*c3 - b3*c2)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (b2*z)*determ_rec + (c2*y)*determ_rec + (z*(a1*b2 - a2*b1)*(b2*c3 - b3*c2))*determ_rec_sq - (y*(a1*c2 - a2*c1)*(b2*c3 - b3*c2))*determ_rec_sq + (x*(b1*c2 - b2*c1)*(b2*c3 - b3*c2))*determ_rec_sq);

            ValueType dx_da2=-((b3*c4 - b4*c3)*determ_rec + ((b1*c3 - b3*c1)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (b3*z)*determ_rec + (c3*y)*determ_rec - (z*(a2*b3 - a3*b2)*(b1*c3 - b3*c1))*determ_rec_sq + (y*(a2*c3 - a3*c2)*(b1*c3 - b3*c1))*determ_rec_sq - (x*(b1*c3 - b3*c1)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_da2=(((b1*c3 - b3*c1)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (x*itk::Math::sqr(b1*c3 - b3*c1))*determ_rec_sq - (z*(a1*b3 - a3*b1)*(b1*c3 - b3*c1))*determ_rec_sq + (y*(a1*c3 - a3*c1)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_da2=((b1*c4 - b4*c1)*determ_rec - ((b1*c3 - b3*c1)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (b1*z)*determ_rec + (c1*y)*determ_rec + (z*(a1*b2 - a2*b1)*(b1*c3 - b3*c1))*determ_rec_sq - (y*(a1*c2 - a2*c1)*(b1*c3 - b3*c1))*determ_rec_sq + (x*(b1*c2 - b2*c1)*(b1*c3 - b3*c1))*determ_rec_sq);

            ValueType dx_da3=((b2*c4 - b4*c2)*determ_rec + ((b1*c2 - b2*c1)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (b2*z)*determ_rec + (c2*y)*determ_rec - (z*(a2*b3 - a3*b2)*(b1*c2 - b2*c1))*determ_rec_sq + (y*(a2*c3 - a3*c2)*(b1*c2 - b2*c1))*determ_rec_sq - (x*(b1*c2 - b2*c1)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_da3=-((b1*c4 - b4*c1)*determ_rec + ((b1*c2 - b2*c1)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (b1*z)*determ_rec + (c1*y)*determ_rec - (z*(a1*b3 - a3*b1)*(b1*c2 - b2*c1))*determ_rec_sq + (y*(a1*c3 - a3*c1)*(b1*c2 - b2*c1))*determ_rec_sq - (x*(b1*c2 - b2*c1)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_da3=(((b1*c2 - b2*c1)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (x*itk::Math::sqr(b1*c2 - b2*c1))*determ_rec_sq - (z*(a1*b2 - a2*b1)*(b1*c2 - b2*c1))*determ_rec_sq + (y*(a1*c2 - a2*c1)*(b1*c2 - b2*c1))*determ_rec_sq);

            ValueType dx_da4=-determ_rec*(b2*c3 - b3*c2);
            ValueType dy_da4=determ_rec*(b1*c3 - b3*c1);
            ValueType dz_da4=-determ_rec*(b1*c2 - b2*c1);

            ///////////////////////////////////////////////////////////////////////////////////

            ValueType dx_db1=-(((a2*c3 - a3*c2)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq + (y*itk::Math::sqr(a2*c3 - a3*c2))*determ_rec_sq - (z*(a2*b3 - a3*b2)*(a2*c3 - a3*c2))*determ_rec_sq - (x*(a2*c3 - a3*c2)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_db1=-((a3*c4 - a4*c3)*determ_rec - ((a2*c3 - a3*c2)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (a3*z)*determ_rec + (c3*x)*determ_rec + (z*(a1*b3 - a3*b1)*(a2*c3 - a3*c2))*determ_rec_sq - (y*(a1*c3 - a3*c1)*(a2*c3 - a3*c2))*determ_rec_sq + (x*(a2*c3 - a3*c2)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_db1=((a2*c4 - a4*c2)*determ_rec - ((a2*c3 - a3*c2)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (a2*z)*determ_rec + (c2*x)*determ_rec + (z*(a1*b2 - a2*b1)*(a2*c3 - a3*c2))*determ_rec_sq - (y*(a1*c2 - a2*c1)*(a2*c3 - a3*c2))*determ_rec_sq + (x*(a2*c3 - a3*c2)*(b1*c2 - b2*c1))*determ_rec_sq);

            ValueType dx_db2=((a3*c4 - a4*c3)*determ_rec + ((a1*c3 - a3*c1)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (a3*z)*determ_rec + (c3*x)*determ_rec - (z*(a2*b3 - a3*b2)*(a1*c3 - a3*c1))*determ_rec_sq + (y*(a1*c3 - a3*c1)*(a2*c3 - a3*c2))*determ_rec_sq - (x*(a1*c3 - a3*c1)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_db2=-(((a1*c3 - a3*c1)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq + (y*itk::Math::sqr(a1*c3 - a3*c1))*determ_rec_sq - (z*(a1*b3 - a3*b1)*(a1*c3 - a3*c1))*determ_rec_sq - (x*(a1*c3 - a3*c1)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_db2=-((a1*c4 - a4*c1)*determ_rec - ((a1*c3 - a3*c1)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (a1*z)*determ_rec + (c1*x)*determ_rec + (z*(a1*b2 - a2*b1)*(a1*c3 - a3*c1))*determ_rec_sq - (y*(a1*c2 - a2*c1)*(a1*c3 - a3*c1))*determ_rec_sq + (x*(a1*c3 - a3*c1)*(b1*c2 - b2*c1))*determ_rec_sq);

            ValueType dx_db3=-((a2*c4 - a4*c2)*determ_rec + ((a1*c2 - a2*c1)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (a2*z)*determ_rec + (c2*x)*determ_rec - (z*(a2*b3 - a3*b2)*(a1*c2 - a2*c1))*determ_rec_sq + (y*(a1*c2 - a2*c1)*(a2*c3 - a3*c2))*determ_rec_sq - (x*(a1*c2 - a2*c1)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_db3=((a1*c4 - a4*c1)*determ_rec + ((a1*c2 - a2*c1)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (a1*z)*determ_rec + (c1*x)*determ_rec - (z*(a1*b3 - a3*b1)*(a1*c2 - a2*c1))*determ_rec_sq + (y*(a1*c2 - a2*c1)*(a1*c3 - a3*c1))*determ_rec_sq - (x*(a1*c2 - a2*c1)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_db3=-(((a1*c2 - a2*c1)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq + (y*itk::Math::sqr(a1*c2 - a2*c1))*determ_rec_sq - (z*(a1*b2 - a2*b1)*(a1*c2 - a2*c1))*determ_rec_sq - (x*(a1*c2 - a2*c1)*(b1*c2 - b2*c1))*determ_rec_sq);

            ValueType dx_db4=determ_rec*(a2*c3 - a3*c2);
            ValueType dy_db4=-determ_rec*(a1*c3 - a3*c1);
            ValueType dz_db4=determ_rec*(a1*c2 - a2*c1);

            //////////////////////////////////////////////////////////////////////////////////

            ValueType dx_dc1=(((a2*b3 - a3*b2)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (z*itk::Math::sqr(a2*b3 - a3*b2))*determ_rec_sq + (y*(a2*b3 - a3*b2)*(a2*c3 - a3*c2))*determ_rec_sq - (x*(a2*b3 - a3*b2)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_dc1=((a3*b4 - a4*b3)*determ_rec - ((a2*b3 - a3*b2)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (a3*y)*determ_rec + (b3*x)*determ_rec + (z*(a1*b3 - a3*b1)*(a2*b3 - a3*b2))*determ_rec_sq - (y*(a2*b3 - a3*b2)*(a1*c3 - a3*c1))*determ_rec_sq + (x*(a2*b3 - a3*b2)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_dc1=-((a2*b4 - a4*b2)*determ_rec - ((a2*b3 - a3*b2)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (a2*y)*determ_rec + (b2*x)*determ_rec + (z*(a1*b2 - a2*b1)*(a2*b3 - a3*b2))*determ_rec_sq - (y*(a2*b3 - a3*b2)*(a1*c2 - a2*c1))*determ_rec_sq + (x*(a2*b3 - a3*b2)*(b1*c2 - b2*c1))*determ_rec_sq);

            ValueType dx_dc2=-((a3*b4 - a4*b3)*determ_rec + ((a1*b3 - a3*b1)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (a3*y)*determ_rec + (b3*x)*determ_rec - (z*(a1*b3 - a3*b1)*(a2*b3 - a3*b2))*determ_rec_sq + (y*(a1*b3 - a3*b1)*(a2*c3 - a3*c2))*determ_rec_sq - (x*(a1*b3 - a3*b1)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_dc2=(((a1*b3 - a3*b1)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (z*itk::Math::sqr(a1*b3 - a3*b1))*determ_rec_sq + (y*(a1*b3 - a3*b1)*(a1*c3 - a3*c1))*determ_rec_sq - (x*(a1*b3 - a3*b1)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_dc2=((a1*b4 - a4*b1)*determ_rec - ((a1*b3 - a3*b1)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (a1*y)*determ_rec + (b1*x)*determ_rec + (z*(a1*b2 - a2*b1)*(a1*b3 - a3*b1))*determ_rec_sq - (y*(a1*b3 - a3*b1)*(a1*c2 - a2*c1))*determ_rec_sq + (x*(a1*b3 - a3*b1)*(b1*c2 - b2*c1))*determ_rec_sq);

            ValueType dx_dc3=((a2*b4 - a4*b2)*determ_rec + ((a1*b2 - a2*b1)*(a2*b3*c4 - a2*b4*c3 - a3*b2*c4 + a3*b4*c2 + a4*b2*c3 - a4*b3*c2))*determ_rec_sq - (a2*y)*determ_rec + (b2*x)*determ_rec - (z*(a1*b2 - a2*b1)*(a2*b3 - a3*b2))*determ_rec_sq + (y*(a1*b2 - a2*b1)*(a2*c3 - a3*c2))*determ_rec_sq - (x*(a1*b2 - a2*b1)*(b2*c3 - b3*c2))*determ_rec_sq);
            ValueType dy_dc3=-((a1*b4 - a4*b1)*determ_rec + ((a1*b2 - a2*b1)*(a1*b3*c4 - a1*b4*c3 - a3*b1*c4 + a3*b4*c1 + a4*b1*c3 - a4*b3*c1))*determ_rec_sq - (a1*y)*determ_rec + (b1*x)*determ_rec - (z*(a1*b2 - a2*b1)*(a1*b3 - a3*b1))*determ_rec_sq + (y*(a1*b2 - a2*b1)*(a1*c3 - a3*c1))*determ_rec_sq - (x*(a1*b2 - a2*b1)*(b1*c3 - b3*c1))*determ_rec_sq);
            ValueType dz_dc3=(((a1*b2 - a2*b1)*(a1*b2*c4 - a1*b4*c2 - a2*b1*c4 + a2*b4*c1 + a4*b1*c2 - a4*b2*c1))*determ_rec_sq - (z*itk::Math::sqr(a1*b2 - a2*b1))*determ_rec_sq + (y*(a1*b2 - a2*b1)*(a1*c2 - a2*c1))*determ_rec_sq - (x*(a1*b2 - a2*b1)*(b1*c2 - b2*c1))*determ_rec_sq);

            ValueType dx_dc4=-determ_rec*(a2*b3 - a3*b2);
            ValueType dy_dc4=determ_rec*(a1*b3 - a3*b1);
            ValueType dz_dc4=-determ_rec*(a1*b2 - a2*b1);



/*
            ValueType dx_da1 = (((b2 * c3 - b3 * c2) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (x * itk::Math::sqr(b2 * c3 - b3 * c2)) * determinantRecSq - (z * (a2 * b3 - a3 * b2) * (b2 * c3 - b3 * c2)) * determinantRecSq + (y * (a2 * c3 - a3 * c2) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_da1 = ((b3 * c4 - b4 * c3) * m_DeterminantRec - ((b2 * c3 - b3 * c2) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (b3 * z) * m_DeterminantRec + (c3 * y) * m_DeterminantRec + (z * (a1 * b3 - a3 * b1) * (b2 * c3 - b3 * c2)) * determinantRecSq - (y * (a1 * c3 - a3 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq + (x * (b1 * c3 - b3 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dz_da1 = -((b2 * c4 - b4 * c2) * m_DeterminantRec - ((b2 * c3 - b3 * c2) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (b2 * z) * m_DeterminantRec + (c2 * y) * m_DeterminantRec + (z * (a1 * b2 - a2 * b1) * (b2 * c3 - b3 * c2)) * determinantRecSq - (y * (a1 * c2 - a2 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq + (x * (b1 * c2 - b2 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq);

            ValueType dx_da2 = -((b3 * c4 - b4 * c3) * m_DeterminantRec + ((b1 * c3 - b3 * c1) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (b3 * z) * m_DeterminantRec + (c3 * y) * m_DeterminantRec - (z * (a2 * b3 - a3 * b2) * (b1 * c3 - b3 * c1)) * determinantRecSq + (y * (a2 * c3 - a3 * c2) * (b1 * c3 - b3 * c1)) * determinantRecSq - (x * (b1 * c3 - b3 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_da2 = (((b1 * c3 - b3 * c1) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (x * itk::Math::sqr(b1 * c3 - b3 * c1)) * determinantRecSq - (z * (a1 * b3 - a3 * b1) * (b1 * c3 - b3 * c1)) * determinantRecSq + (y * (a1 * c3 - a3 * c1) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_da2 = ((b1 * c4 - b4 * c1) * m_DeterminantRec - ((b1 * c3 - b3 * c1) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (b1 * z) * m_DeterminantRec + (c1 * y) * m_DeterminantRec + (z * (a1 * b2 - a2 * b1) * (b1 * c3 - b3 * c1)) * determinantRecSq - (y * (a1 * c2 - a2 * c1) * (b1 * c3 - b3 * c1)) * determinantRecSq + (x * (b1 * c2 - b2 * c1) * (b1 * c3 - b3 * c1)) * determinantRecSq);

            ValueType dx_da3 = ((b2 * c4 - b4 * c2) * m_DeterminantRec + ((b1 * c2 - b2 * c1) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (b2 * z) * m_DeterminantRec + (c2 * y) * m_DeterminantRec - (z * (a2 * b3 - a3 * b2) * (b1 * c2 - b2 * c1)) * determinantRecSq + (y * (a2 * c3 - a3 * c2) * (b1 * c2 - b2 * c1)) * determinantRecSq - (x * (b1 * c2 - b2 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_da3 = -((b1 * c4 - b4 * c1) * m_DeterminantRec + ((b1 * c2 - b2 * c1) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (b1 * z) * m_DeterminantRec + (c1 * y) * m_DeterminantRec - (z * (a1 * b3 - a3 * b1) * (b1 * c2 - b2 * c1)) * determinantRecSq + (y * (a1 * c3 - a3 * c1) * (b1 * c2 - b2 * c1)) * determinantRecSq - (x * (b1 * c2 - b2 * c1) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_da3 = (((b1 * c2 - b2 * c1) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (x * itk::Math::sqr(b1 * c2 - b2 * c1)) * determinantRecSq - (z * (a1 * b2 - a2 * b1) * (b1 * c2 - b2 * c1)) * determinantRecSq + (y * (a1 * c2 - a2 * c1) * (b1 * c2 - b2 * c1)) * determinantRecSq);

            ValueType dx_da4 = -m_DeterminantRec * (b2 * c3 - b3 * c2);
            ValueType dy_da4 = m_DeterminantRec * (b1 * c3 - b3 * c1);
            ValueType dz_da4 = -m_DeterminantRec * (b1 * c2 - b2 * c1);

            /////////////////////////////////////////////////////////////////////////////////

            ValueType dx_db1 = -(((a2 * c3 - a3 * c2) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq + (y * itk::Math::sqr(a2 * c3 - a3 * c2)) * determinantRecSq - (z * (a2 * b3 - a3 * b2) * (a2 * c3 - a3 * c2)) * determinantRecSq - (x * (a2 * c3 - a3 * c2) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_db1 = -((a3 * c4 - a4 * c3) * m_DeterminantRec - ((a2 * c3 - a3 * c2) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (a3 * z) * m_DeterminantRec + (c3 * x) * m_DeterminantRec + (z * (a1 * b3 - a3 * b1) * (a2 * c3 - a3 * c2)) * determinantRecSq - (y * (a1 * c3 - a3 * c1) * (a2 * c3 - a3 * c2)) * determinantRecSq + (x * (a2 * c3 - a3 * c2) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_db1 = ((a2 * c4 - a4 * c2) * m_DeterminantRec - ((a2 * c3 - a3 * c2) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (a2 * z) * m_DeterminantRec + (c2 * x) * m_DeterminantRec + (z * (a1 * b2 - a2 * b1) * (a2 * c3 - a3 * c2)) * determinantRecSq - (y * (a1 * c2 - a2 * c1) * (a2 * c3 - a3 * c2)) * determinantRecSq + (x * (a2 * c3 - a3 * c2) * (b1 * c2 - b2 * c1)) * determinantRecSq);

            ValueType dx_db2 = ((a3 * c4 - a4 * c3) * m_DeterminantRec + ((a1 * c3 - a3 * c1) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (a3 * z) * m_DeterminantRec + (c3 * x) * m_DeterminantRec - (z * (a2 * b3 - a3 * b2) * (a1 * c3 - a3 * c1)) * determinantRecSq + (y * (a1 * c3 - a3 * c1) * (a2 * c3 - a3 * c2)) * determinantRecSq - (x * (a1 * c3 - a3 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_db2 = -(((a1 * c3 - a3 * c1) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq + (y * itk::Math::sqr(a1 * c3 - a3 * c1)) * determinantRecSq - (z * (a1 * b3 - a3 * b1) * (a1 * c3 - a3 * c1)) * determinantRecSq - (x * (a1 * c3 - a3 * c1) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_db2 = -((a1 * c4 - a4 * c1) * m_DeterminantRec - ((a1 * c3 - a3 * c1) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (a1 * z) * m_DeterminantRec + (c1 * x) * m_DeterminantRec + (z * (a1 * b2 - a2 * b1) * (a1 * c3 - a3 * c1)) * determinantRecSq - (y * (a1 * c2 - a2 * c1) * (a1 * c3 - a3 * c1)) * determinantRecSq + (x * (a1 * c3 - a3 * c1) * (b1 * c2 - b2 * c1)) * determinantRecSq);

            ValueType dx_db3 = -((a2 * c4 - a4 * c2) * m_DeterminantRec + ((a1 * c2 - a2 * c1) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (a2 * z) * m_DeterminantRec + (c2 * x) * m_DeterminantRec - (z * (a2 * b3 - a3 * b2) * (a1 * c2 - a2 * c1)) * determinantRecSq + (y * (a1 * c2 - a2 * c1) * (a2 * c3 - a3 * c2)) * determinantRecSq - (x * (a1 * c2 - a2 * c1) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_db3 = ((a1 * c4 - a4 * c1) * m_DeterminantRec + ((a1 * c2 - a2 * c1) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (a1 * z) * m_DeterminantRec + (c1 * x) * m_DeterminantRec - (z * (a1 * b3 - a3 * b1) * (a1 * c2 - a2 * c1)) * determinantRecSq + (y * (a1 * c2 - a2 * c1) * (a1 * c3 - a3 * c1)) * determinantRecSq - (x * (a1 * c2 - a2 * c1) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_db3 = -(((a1 * c2 - a2 * c1) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq + (y * itk::Math::sqr(a1 * c2 - a2 * c1)) * determinantRecSq - (z * (a1 * b2 - a2 * b1) * (a1 * c2 - a2 * c1)) * determinantRecSq - (x * (a1 * c2 - a2 * c1) * (b1 * c2 - b2 * c1)) * determinantRecSq);

            ValueType dx_db4 = m_DeterminantRec * (a2 * c3 - a3 * c2);
            ValueType dy_db4 = -m_DeterminantRec * (a1 * c3 - a3 * c1);
            ValueType dz_db4 = m_DeterminantRec * (a1 * c2 - a2 * c1);

            /////////////////////////////////////////////////////////////////////////////////

            ValueType dx_dc1 = (((a2 * b3 - a3 * b2) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (z * itk::Math::sqr(a2 * b3 - a3 * b2)) * determinantRecSq + (y * (a2 * b3 - a3 * b2) * (a2 * c3 - a3 * c2)) * determinantRecSq - (x * (a2 * b3 - a3 * b2) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_dc1 = ((a3 * b4 - a4 * b3) * m_DeterminantRec - ((a2 * b3 - a3 * b2) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (a3 * y) * m_DeterminantRec + (b3 * x) * m_DeterminantRec + (z * (a1 * b3 - a3 * b1) * (a2 * b3 - a3 * b2)) * determinantRecSq - (y * (a2 * b3 - a3 * b2) * (a1 * c3 - a3 * c1)) * determinantRecSq + (x * (a2 * b3 - a3 * b2) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_dc1 = -((a2 * b4 - a4 * b2) * m_DeterminantRec - ((a2 * b3 - a3 * b2) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (a2 * y) * m_DeterminantRec + (b2 * x) * m_DeterminantRec + (z * (a1 * b2 - a2 * b1) * (a2 * b3 - a3 * b2)) * determinantRecSq - (y * (a2 * b3 - a3 * b2) * (a1 * c2 - a2 * c1)) * determinantRecSq + (x * (a2 * b3 - a3 * b2) * (b1 * c2 - b2 * c1)) * determinantRecSq);

            ValueType dx_dc2 = -((a3 * b4 - a4 * b3) * m_DeterminantRec + ((a1 * b3 - a3 * b1) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (a3 * y) * m_DeterminantRec + (b3 * x) * m_DeterminantRec - (z * (a1 * b3 - a3 * b1) * (a2 * b3 - a3 * b2)) * determinantRecSq + (y * (a1 * b3 - a3 * b1) * (a2 * c3 - a3 * c2)) * determinantRecSq - (x * (a1 * b3 - a3 * b1) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_dc2 = (((a1 * b3 - a3 * b1) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (z * itk::Math::sqr(a1 * b3 - a3 * b1)) * determinantRecSq + (y * (a1 * b3 - a3 * b1) * (a1 * c3 - a3 * c1)) * determinantRecSq - (x * (a1 * b3 - a3 * b1) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_dc2 = ((a1 * b4 - a4 * b1) * m_DeterminantRec - ((a1 * b3 - a3 * b1) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (a1 * y) * m_DeterminantRec + (b1 * x) * m_DeterminantRec + (z * (a1 * b2 - a2 * b1) * (a1 * b3 - a3 * b1)) * determinantRecSq - (y * (a1 * b3 - a3 * b1) * (a1 * c2 - a2 * c1)) * determinantRecSq + (x * (a1 * b3 - a3 * b1) * (b1 * c2 - b2 * c1)) * determinantRecSq);

            ValueType dx_dc3 = ((a2 * b4 - a4 * b2) * m_DeterminantRec + ((a1 * b2 - a2 * b1) * (a2 * b3 * c4 - a2 * b4 * c3 - a3 * b2 * c4 + a3 * b4 * c2 + a4 * b2 * c3 - a4 * b3 * c2)) * determinantRecSq - (a2 * y) * m_DeterminantRec + (b2 * x) * m_DeterminantRec - (z * (a1 * b2 - a2 * b1) * (a2 * b3 - a3 * b2)) * determinantRecSq + (y * (a1 * b2 - a2 * b1) * (a2 * c3 - a3 * c2)) * determinantRecSq - (x * (a1 * b2 - a2 * b1) * (b2 * c3 - b3 * c2)) * determinantRecSq);
            ValueType dy_dc3 = -((a1 * b4 - a4 * b1) * m_DeterminantRec + ((a1 * b2 - a2 * b1) * (a1 * b3 * c4 - a1 * b4 * c3 - a3 * b1 * c4 + a3 * b4 * c1 + a4 * b1 * c3 - a4 * b3 * c1)) * determinantRecSq - (a1 * y) * m_DeterminantRec + (b1 * x) * m_DeterminantRec - (z * (a1 * b2 - a2 * b1) * (a1 * b3 - a3 * b1)) * determinantRecSq + (y * (a1 * b2 - a2 * b1) * (a1 * c3 - a3 * c1)) * determinantRecSq - (x * (a1 * b2 - a2 * b1) * (b1 * c3 - b3 * c1)) * determinantRecSq);
            ValueType dz_dc3 = (((a1 * b2 - a2 * b1) * (a1 * b2 * c4 - a1 * b4 * c2 - a2 * b1 * c4 + a2 * b4 * c1 + a4 * b1 * c2 - a4 * b2 * c1)) * determinantRecSq - (z * itk::Math::sqr(a1 * b2 - a2 * b1)) * determinantRecSq + (y * (a1 * b2 - a2 * b1) * (a1 * c2 - a2 * c1)) * determinantRecSq - (x * (a1 * b2 - a2 * b1) * (b1 * c2 - b2 * c1)) * determinantRecSq);

            ValueType dx_dc4 = -m_DeterminantRec * (a2 * b3 - a3 * b2);
            ValueType dy_dc4 = m_DeterminantRec * (a1 * b3 - a3 * b1);
            ValueType dz_dc4 = -m_DeterminantRec * (a1 * b2 - a2 * b1);
*/
            /////////////////////////////////////////////////////////////////////////////////
            //out[0].Fill(0);
            //out[1].Fill(0);
            //out[2].Fill(0);
            
            out[0][0] = dx_da1;
            out[0][1] = dx_da2;
            out[0][2] = dx_da3;
            out[0][3] = dx_da4;

            out[0][4] = dx_db1;
            out[0][5] = dx_db2;
            out[0][6] = dx_db3;
            out[0][7] = dx_db4;

            out[0][8] = dx_dc1;
            out[0][9] = dx_dc2;
            out[0][10] = dx_dc3;
            out[0][11] = dx_dc4;

            out[1][0] = dy_da1;
            out[1][1] = dy_da2;
            out[1][2] = dy_da3;
            out[1][3] = dy_da4;

            out[1][4] = dy_db1;
            out[1][5] = dy_db2;
            out[1][6] = dy_db3;
            out[1][7] = dy_db4;

            out[1][8] = dy_dc1;
            out[1][9] = dy_dc2;
            out[1][10] = dy_dc3;
            out[1][11] = dy_dc4;

            out[2][0] = dz_da1;
            out[2][1] = dz_da2;
            out[2][2] = dz_da3;
            out[2][3] = dz_da4;

            out[2][4] = dz_db1;
            out[2][5] = dz_db2;
            out[2][6] = dz_db3;
            out[2][7] = dz_db4;

            out[2][8] = dz_dc1;
            out[2][9] = dz_dc2;
            out[2][10] = dz_dc3;
            out[2][11] = dz_dc4;
        }

        //DV/DT = DV/DS * DS/DT
        template <typename OutArrayType>
        inline void TotalDiffInverse(Point p, itk::CovariantVector<ValueType, Dim> spatialGrad, OutArrayType& out, ValueType weight, bool addDerivative = true) {
            ParamArrayType d[Dim];

            DiffInverse(p, d);

            for(unsigned int i = 0; i < NParams; ++i) {
                ValueType acc = 0;

                for(unsigned int j = 0; j < Dim; ++j) {
                    acc += weight * spatialGrad[j] * d[j][i];
                }

                if(addDerivative)
                    out[i] += acc;
                else
                    out[i] = acc;
            }
        }
        protected:
            ValueType m_DeterminantRec;
    };

}

#endif
