#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>

#define N 1000

// Compare the matrices mat1 and mat2, and return 1 if they are equal,
// 0 otherwise
int compare_matrices(int mat1[N][N], int mat2[N][N])
{
	int i, j;
	for (i = 0; i < N; ++i) for (j = 0; j < N; ++j)
		if (mat1[i][j] != mat2[i][j])
			return 0;
	return 1;
}

// Standard version
void version1(int mat1[N][N], int mat2[N][N], int result[N][N])
{
	int i, j, k;
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			for (k = 0; k < N; ++k)
				result[i][j] += mat1[i][k] * mat2[k][j];
		}
	}
}

// SSE version
void SSE(int mat1[N][N], int mat2[N][N], int result[N][N])
{
	__m128i sum_vec = _mm_setzero_si128();
	__m128i mat1_vec = _mm_setzero_si128();
	__m128i  mat2_vec = _mm_setzero_si128();

	int i, j, k;
	for (i = 0; i < N; i += 4)
	{
		for (j = 0; j < N; ++j)
		{
			mat2_vec = _mm_setr_epi32(mat2[i][j], mat2[i + 1][j], mat2[i + 2][j], mat2[i + 3][j]);
			for (k = 0; k < N; ++k)
			{
				mat1_vec = _mm_loadu_si128((const __m128i*)&mat1[k][i]);
				sum_vec = _mm_add_epi32(sum_vec, _mm_mullo_epi16(mat1_vec, mat2_vec));
				result[k][j] += _mm_extract_epi32(sum_vec, 0) + _mm_extract_epi32(sum_vec, 1) + _mm_extract_epi32(sum_vec, 2) + _mm_extract_epi32(sum_vec, 3);
				sum_vec = _mm_setzero_si128();
			}
		}
	}
}

// AVX version
void AVX(int mat1[N][N], int mat2[N][N], int result[N][N])
{
	__m256i sum_vec = _mm256_setzero_si256();
	__m256i mat1_vec = _mm256_setzero_si256();
	__m256i mat2_vec = _mm256_setzero_si256();

	int i, j, k;
	for (i = 0; i < N; i += 8)
	{
		for (j = 0; j < N; ++j)
		{
			mat2_vec = _mm256_setr_epi32(mat2[i][j], mat2[i + 1][j], mat2[i + 2][j], mat2[i + 3][j], mat2[i + 4][j], mat2[i + 5][j], mat2[i + 6][j], mat2[i + 7][j]);
			for (k = 0; k < N; ++k)
			{
				
				mat1_vec = _mm256_loadu_si256((const __m256i*)&mat1[k][i]);
				sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(mat1_vec, mat2_vec));
				result[k][j] += _mm256_extract_epi32(sum_vec, 0) + _mm256_extract_epi32(sum_vec, 1) + _mm256_extract_epi32(sum_vec, 2) + _mm256_extract_epi32(sum_vec, 3) + _mm256_extract_epi32(sum_vec, 4) + _mm256_extract_epi32(sum_vec, 5) + _mm256_extract_epi32(sum_vec, 6) + _mm256_extract_epi32(sum_vec, 7);
				sum_vec = _mm256_setzero_si256();
			}
		}
	}
}


// The matrices. mat_ref is used for reference. If the multiplication is done correctly,
// mat_r should equal mat_ref.
int mat_a[N][N], mat_b[N][N], mat_r_SSE[N][N], mat_r_AVX[N][N], mat_ref[N][N];

// Call this before performing the operation (and do *not* include the time to
// return from this function in your measurements). It fills mat_a and mat_b with
// random integer values in the range [0..9].
void init_matrices()
{
	int i, j;
	srand(0xBADB0LL);
	for (i = 0; i < N; ++i) for (j = 0; j < N; ++j)
	{
		mat_a[i][j] = rand() % 10;
		mat_b[i][j] = rand() % 10;
		mat_r_SSE[i][j] = 0;
		mat_r_AVX[i][j] = 0;
		mat_ref[i][j] = 0;
	}
}

int main(void)
{
	clock_t t0, t1;

	// Initialize the matrices
	init_matrices();


	// Check that mat_r is correct. For this the reference matrix mat_ref is computed
	// using the basic() implementation,
	version1(mat_a, mat_b, mat_ref);

	//SSE

	// Take the time
	t0 = clock();
	// Run the algorithm
	SSE(mat_a, mat_b, mat_r_SSE);
	// Take the time again
	t1 = clock();

	printf("SSE Finished in %lf seconds.\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
	//  and then mat_r is compared to mat_ref.
	if (!compare_matrices(mat_r_SSE, mat_ref))
		printf("Error: mat_r does not match the reference matrix!\n");
	


	//AVX

	// Take the time
	t0 = clock();
	// Run the algorithm
	AVX(mat_a, mat_b, mat_r_AVX);
	// Take the time again
	t1 = clock();

	printf("AVX Finished in %lf seconds.\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
	//  and then mat_r is compared to mat_ref.
	if (!compare_matrices(mat_r_AVX, mat_ref))
		printf("Error: mat_r does not match the reference matrix!\n");
	

	// If using Visual Studio, do not close the console window immediately
#ifdef _MSC_VER
	system("pause");
#endif

	return 0;
}