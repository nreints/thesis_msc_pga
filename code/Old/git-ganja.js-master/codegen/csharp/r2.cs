// 3D Projective Geometric Algebra
// Written by a generator written by enki.
using System;
using System.Text;
using static R2.R2; // static variable acces

namespace R2
{
	public class R2
	{
		// just for debug and print output, the basis names
		public static string[] _basis = new[] { "1","e1","e2","e12" };

		private float[] _mVec = new float[4];

		/// <summary>
		/// Ctor
		/// </summary>
		/// <param name="f"></param>
		/// <param name="idx"></param>
		public R2(float f = 0f, int idx = 0)
		{
			_mVec[idx] = f;
		}

		#region Array Access
		public float this[int idx]
		{
			get { return _mVec[idx]; }
			set { _mVec[idx] = value; }
		}
		#endregion

		#region Overloaded Operators

		/// <summary>
		/// R2.Reverse : res = ~a
		/// Reverse the order of the basis blades.
		/// </summary>
		public static R2 operator ~ (R2 a)
		{
			R2 res = new R2();
			res[0]=a[0];
			res[1]=a[1];
			res[2]=a[2];
			res[3]=-a[3];
			return res;
		}

		/// <summary>
		/// R2.Dual : res = !a
		/// Poincare duality operator.
		/// </summary>
		public static R2 operator ! (R2 a)
		{
			R2 res = new R2();
			res[0]=-a[3];
			res[1]=a[2];
			res[2]=-a[1];
			res[3]=a[0];
			return res;
		}

		/// <summary>
		/// R2.Conjugate : res = a.Conjugate()
		/// Clifford Conjugation
		/// </summary>
		public  R2 Conjugate ()
		{
			R2 res = new R2();
			res[0]=this[0];
			res[1]=-this[1];
			res[2]=-this[2];
			res[3]=-this[3];
			return res;
		}

		/// <summary>
		/// R2.Involute : res = a.Involute()
		/// Main involution
		/// </summary>
		public  R2 Involute ()
		{
			R2 res = new R2();
			res[0]=this[0];
			res[1]=-this[1];
			res[2]=-this[2];
			res[3]=this[3];
			return res;
		}

		/// <summary>
		/// R2.Mul : res = a * b
		/// The geometric product.
		/// </summary>
		public static R2 operator * (R2 a, R2 b)
		{
			R2 res = new R2();
			res[0]=b[0]*a[0]+b[1]*a[1]+b[2]*a[2]-b[3]*a[3];
			res[1]=b[1]*a[0]+b[0]*a[1]-b[3]*a[2]+b[2]*a[3];
			res[2]=b[2]*a[0]+b[3]*a[1]+b[0]*a[2]-b[1]*a[3];
			res[3]=b[3]*a[0]+b[2]*a[1]-b[1]*a[2]+b[0]*a[3];
			return res;
		}

		/// <summary>
		/// R2.Wedge : res = a ^ b
		/// The outer product. (MEET)
		/// </summary>
		public static R2 operator ^ (R2 a, R2 b)
		{
			R2 res = new R2();
			res[0]=b[0]*a[0];
			res[1]=b[1]*a[0]+b[0]*a[1];
			res[2]=b[2]*a[0]+b[0]*a[2];
			res[3]=b[3]*a[0]+b[2]*a[1]-b[1]*a[2]+b[0]*a[3];
			return res;
		}

		/// <summary>
		/// R2.Vee : res = a & b
		/// The regressive product. (JOIN)
		/// </summary>
		public static R2 operator & (R2 a, R2 b)
		{
			R2 res = new R2();
			res[3]=1*(a[3]*b[3]);
			res[2]=-1*(a[2]*-1*b[3]+a[3]*b[2]*-1);
			res[1]=1*(a[1]*b[3]+a[3]*b[1]);
			res[0]=1*(a[0]*b[3]+a[1]*b[2]*-1-a[2]*-1*b[1]+a[3]*b[0]);
			return res;
		}

		/// <summary>
		/// R2.Dot : res = a | b
		/// The inner product.
		/// </summary>
		public static R2 operator | (R2 a, R2 b)
		{
			R2 res = new R2();
			res[0]=b[0]*a[0]+b[1]*a[1]+b[2]*a[2]-b[3]*a[3];
			res[1]=b[1]*a[0]+b[0]*a[1]-b[3]*a[2]+b[2]*a[3];
			res[2]=b[2]*a[0]+b[3]*a[1]+b[0]*a[2]-b[1]*a[3];
			res[3]=b[3]*a[0]+b[0]*a[3];
			return res;
		}

		/// <summary>
		/// R2.Add : res = a + b
		/// Multivector addition
		/// </summary>
		public static R2 operator + (R2 a, R2 b)
		{
			R2 res = new R2();
			res[0] = a[0]+b[0];
			res[1] = a[1]+b[1];
			res[2] = a[2]+b[2];
			res[3] = a[3]+b[3];
			return res;
		}

		/// <summary>
		/// R2.Sub : res = a - b
		/// Multivector subtraction
		/// </summary>
		public static R2 operator - (R2 a, R2 b)
		{
			R2 res = new R2();
			res[0] = a[0]-b[0];
			res[1] = a[1]-b[1];
			res[2] = a[2]-b[2];
			res[3] = a[3]-b[3];
			return res;
		}

		/// <summary>
		/// R2.smul : res = a * b
		/// scalar/multivector multiplication
		/// </summary>
		public static R2 operator * (float a, R2 b)
		{
			R2 res = new R2();
			res[0] = a*b[0];
			res[1] = a*b[1];
			res[2] = a*b[2];
			res[3] = a*b[3];
			return res;
		}

		/// <summary>
		/// R2.muls : res = a * b
		/// multivector/scalar multiplication
		/// </summary>
		public static R2 operator * (R2 a, float b)
		{
			R2 res = new R2();
			res[0] = a[0]*b;
			res[1] = a[1]*b;
			res[2] = a[2]*b;
			res[3] = a[3]*b;
			return res;
		}

		/// <summary>
		/// R2.sadd : res = a + b
		/// scalar/multivector addition
		/// </summary>
		public static R2 operator + (float a, R2 b)
		{
			R2 res = new R2();
			res[0] = a+b[0];
			res[1] = b[1];
			res[2] = b[2];
			res[3] = b[3];
			return res;
		}

		/// <summary>
		/// R2.adds : res = a + b
		/// multivector/scalar addition
		/// </summary>
		public static R2 operator + (R2 a, float b)
		{
			R2 res = new R2();
			res[0] = a[0]+b;
			res[1] = a[1];
			res[2] = a[2];
			res[3] = a[3];
			return res;
		}

		/// <summary>
		/// R2.ssub : res = a - b
		/// scalar/multivector subtraction
		/// </summary>
		public static R2 operator - (float a, R2 b)
		{
			R2 res = new R2();
			res[0] = a-b[0];
			res[1] = -b[1];
			res[2] = -b[2];
			res[3] = -b[3];
			return res;
		}

		/// <summary>
		/// R2.subs : res = a - b
		/// multivector/scalar subtraction
		/// </summary>
		public static R2 operator - (R2 a, float b)
		{
			R2 res = new R2();
			res[0] = a[0]-b;
			res[1] = a[1];
			res[2] = a[2];
			res[3] = a[3];
			return res;
		}

		#endregion

                /// <summary>
                /// R2.norm()
                /// Calculate the Euclidean norm. (strict positive).
                /// </summary>
		public float norm() { return (float) Math.Sqrt(Math.Abs((this*this.Conjugate())[0]));}
		
		/// <summary>
		/// R2.inorm()
		/// Calculate the Ideal norm. (signed)
		/// </summary>
		public float inorm() { return this[1]!=0.0f?this[1]:this[15]!=0.0f?this[15]:(!this).norm();}
		
		/// <summary>
		/// R2.normalized()
		/// Returns a normalized (Euclidean) element.
		/// </summary>
		public R2 normalized() { return this*(1/norm()); }
		
		
		// The basis blades
		public static R2 e1 = new R2(1f, 1);
		public static R2 e2 = new R2(1f, 2);
		public static R2 e12 = new R2(1f, 3);

		
		/// string cast
		public override string ToString()
		{
			var sb = new StringBuilder();
			var n=0;
			for (int i = 0; i < 4; ++i) 
				if (_mVec[i] != 0.0f) {
					sb.Append($"{_mVec[i]}{(i == 0 ? string.Empty : _basis[i])} + ");
					n++;
			        }
			if (n==0) sb.Append("0");
			return sb.ToString().TrimEnd(' ', '+');
		}
	}

	class Program
	{
	        

		static void Main(string[] args)
		{
		
			Console.WriteLine("e1*e1         : "+e1*e1);
			Console.WriteLine("pss           : "+e12);
			Console.WriteLine("pss*pss       : "+e12*e12);

		}
	}
}

