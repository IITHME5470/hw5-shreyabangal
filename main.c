#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NX 100
#define NY 100
#define STEPS 1000
#define OUTPUT_INTERVAL 100

// Dummy prototype for output_error_norm, replace with actual implementation
void output_error_norm(int rank, int size, int nx, int nxglob, double tcurr, double **T, double *Texact)
{
    if (rank == 0) {
        printf("output_error_norm called at t = %f\n", tcurr);
    }
}
// Your get_rhs function
void get_rhs(int nx, int nxglob, int ny, int nyglob, int istglob, int ienglob,
             int jstglob, int jenglob, double dx, double dy,
             double *xleftghost, double *xrightghost, double *ybotghost, double *ytopghost,
             double kdiff, double *x, double *y, double **T, double **rhs)
{
    int i, j;
    double dxsq = dx * dx, dysq = dy * dy;

    for (i = 1; i < nx - 1; i++)
        for (j = 1; j < ny - 1; j++)
            rhs[i][j] = kdiff * (T[i + 1][j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                        kdiff * (T[i][j + 1] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;

    // Left boundary
    i = 0;
    if (istglob == 0)
        for (j = 1; j < ny - 1; j++)
            rhs[i][j] = kdiff * (T[i + 1][j] + xleftghost[j] - 2.0 * T[i][j]) / dxsq +
                        kdiff * (T[i][j + 1] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;

    // Right boundary
    i = nx - 1;
    if (ienglob == nxglob - 1)
        for (j = 1; j < ny - 1; j++)
            rhs[i][j] = 0.0;
    else
        for (j = 1; j < ny - 1; j++)
            rhs[i][j] = kdiff * (xrightghost[j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                        kdiff * (T[i][j + 1] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;

    // Bottom boundary
    j = 0;
    if (jstglob == 0)
        for (i = 1; i < nx - 1; i++)
            rhs[i][j] = 0.0;
    else
        for (i = 1; i < nx - 1; i++)
            rhs[i][j] = kdiff * (T[i + 1][j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                        kdiff * (T[i][j + 1] + ybotghost[i] - 2.0 * T[i][j]) / dysq;

    // Top boundary
    j = ny - 1;
    if (jenglob == nyglob - 1)
        for (i = 1; i < nx - 1; i++)
            rhs[i][j] = 0.0;
    else
        for (i = 1; i < nx - 1; i++)
            rhs[i][j] = kdiff * (T[i + 1][j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                        kdiff * (ytopghost[i] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;

    // Corners
    // Bottom-left
    i = 0; j = 0;
    if (istglob == 0 || jstglob == 0)
        rhs[i][j] = 0.0;
    else
        rhs[i][j] = kdiff * (T[i + 1][j] + xleftghost[j] - 2.0 * T[i][j]) / dxsq +
                    kdiff * (T[i][j + 1] + ybotghost[i] - 2.0 * T[i][j]) / dysq;

    // Bottom-right
    i = nx - 1; j = 0;
    if (ienglob == nxglob - 1 || jstglob == 0)
        rhs[i][j] = 0.0;
    else
        rhs[i][j] = kdiff * (xrightghost[j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                    kdiff * (T[i][j + 1] + ybotghost[i] - 2.0 * T[i][j]) / dysq;

    // Top-left
    i = 0; j = ny - 1;
    if (istglob == 0 || jenglob == nyglob - 1)
        rhs[i][j] = 0.0;
    else
        rhs[i][j] = kdiff * (T[i + 1][j] + xleftghost[j] - 2.0 * T[i][j]) / dxsq +
                    kdiff * (ytopghost[i] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;

    // Top-right
    i = nx - 1; j = ny - 1;
    if (ienglob == nxglob - 1 || jenglob == nyglob - 1)
        rhs[i][j] = 0.0;
    else
        rhs[i][j] = kdiff * (xrightghost[j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                    kdiff * (ytopghost[i] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;
}

// Halo exchange in X
void halo_exchange_2d_x(int rank, int rank_x, int rank_y, int size, int px, int py,
                        int nx, int ny, int nxglob, int nyglob, double *x, double *y,
                        double **T, double *xleftghost, double *xrightghost,
                        double *sendbuf_x, double *recvbuf_x)
{
    MPI_Status status;
    int left_nb = (rank_x == 0) ? MPI_PROC_NULL : rank - 1;
    int right_nb = (rank_x == px - 1) ? MPI_PROC_NULL : rank + 1;

    for (int j = 0; j < ny; j++)
        sendbuf_x[j] = T[0][j];
    MPI_Recv(recvbuf_x, ny, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD, &status);
    MPI_Send(sendbuf_x, ny, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD);
    for (int j = 0; j < ny; j++)
        xrightghost[j] = recvbuf_x[j];

    for (int j = 0; j < ny; j++)
        sendbuf_x[j] = T[nx - 1][j];
    MPI_Recv(recvbuf_x, ny, MPI_DOUBLE, left_nb, 0, MPI_COMM_WORLD, &status);
    MPI_Send(sendbuf_x, ny, MPI_DOUBLE, right_nb, 0, MPI_COMM_WORLD);
    for (int j = 0; j < ny; j++)
        xleftghost[j] = recvbuf_x[j];
}

// Halo exchange in Y
void halo_exchange_2d_y(int rank, int rank_x, int rank_y, int size, int px, int py,
                        int nx, int ny, int nxglob, int nyglob, double *x, double *y,
                        double **T, double *ybotghost, double *ytopghost,
                        double *sendbuf_y, double *recvbuf_y)
{
    MPI_Status status;
    int bot_nb = (rank_y == 0) ? MPI_PROC_NULL : rank - px;
    int top_nb = (rank_y == py - 1) ? MPI_PROC_NULL : rank + px;

    for (int i = 0; i < nx; i++)
        sendbuf_y[i] = T[i][0];
    MPI_Recv(recvbuf_y, nx, MPI_DOUBLE, top_nb, 0, MPI_COMM_WORLD, &status);
    MPI_Send(sendbuf_y, nx, MPI_DOUBLE, bot_nb, 0, MPI_COMM_WORLD);
    for (int i = 0; i < nx; i++)
        ybotghost[i] = recvbuf_y[i];

    for (int i = 0; i < nx; i++)
        sendbuf_y[i] = T[i][ny - 1];
    MPI_Recv(recvbuf_y, nx, MPI_DOUBLE, bot_nb, 0, MPI_COMM_WORLD, &status);
    MPI_Send(sendbuf_y, nx, MPI_DOUBLE, top_nb, 0, MPI_COMM_WORLD);
    for (int i = 0; i < nx; i++)
        ytopghost[i] = recvbuf_y[i];
}

void write_output(double *T, int nx, int ny, double dx, double dy, int timestep) {
  char filename[256];
  sprintf(filename, "T_x_y_%06d.dat", timestep);
  FILE *fp = fopen(filename, "w");
  if (!fp) {
      perror("Failed to open file for writing");
      exit(1);
  }

  for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
          double x = i * dx;
          double y = j * dy;
          fprintf(fp, "%lf %lf %lf\n", x, y, T[j * nx + i]);
      }
  }
  fclose(fp);
}

// Main function
int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Assume rank 0 reads the input and broadcasts
  FILE *fp;
  int nx = NX, ny = NY;
  double dx = 1.0 / (nx - 1), dy = 1.0 / (ny - 1);
  double *T = (double *)calloc(nx * ny, sizeof(double));
  double *Tnew = (double *)calloc(nx * ny, sizeof(double));

  // Read input2d.in to initialize T
  if (rank == 0) {
      fp = fopen("input2d.in", "r");
      if (!fp) {
          perror("Cannot open input file");
          MPI_Abort(MPI_COMM_WORLD, 1);
      }
      for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
              fscanf(fp, "%lf", &T[j * nx + i]);
          }
      }
      fclose(fp);
  }

  // Broadcast initial T to all processes
  MPI_Bcast(T, nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int step = 0; step <= STEPS; step++) {
      for (int j = 1; j < ny - 1; j++) {
          for (int i = 1; i < nx - 1; i++) {
              Tnew[j * nx + i] = 0.25 * (
                  T[j * nx + (i - 1)] + T[j * nx + (i + 1)] +
                  T[(j - 1) * nx + i] + T[(j + 1) * nx + i]
              );
          }
      }

      // Swap pointers
      double *temp = T;
      T = Tnew;
      Tnew = temp;

      if (step % OUTPUT_INTERVAL == 0 && rank == 0) {
          write_output(T, nx, ny, dx, dy, step);
      }
  }

  free(T);
  free(Tnew);
  MPI_Finalize();
  return 0;
}
