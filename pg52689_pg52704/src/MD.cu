/*
 MD.c - a simple molecular dynamics program for simulating real gas properties of Lennard-Jones particles.
 
 Copyright (C) 2016  Jonathan J. Foley IV, Chelsea Sweet, Oyewumi Akinfenwa
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 Electronic Contact:  foleyj10@wpunj.edu
 Mail Contact:   Prof. Jonathan Foley
 Department of Chemistry, William Paterson University
 300 Pompton Road
 Wayne NJ 07470
 
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Number of particles
int N;

//  Lennard-Jones parameters in natural units!
double sigma = 1.;
double epsilon = 1.;
double m = 1.;
double kB = 1.;

double NA = 6.022140857e23;
double kBSI = 1.38064852e-23;  // m^2*kg/(s^2*K)

//  Size of box, which will be specified in natural units
double L;

//  Initial Temperature in Natural Units
double Tinit;  //2;
//  Vectors!
//
const int MAXPART=5001;

// Position
double rx[MAXPART];
double ry[MAXPART];
double rz[MAXPART];

// Velocity
double vx[MAXPART];
double vy[MAXPART];
double vz[MAXPART];

// Acceleration
double ax[MAXPART];
double ay[MAXPART];
double az[MAXPART];

// Force
double Fx[MAXPART];
double Fy[MAXPART];
double Fz[MAXPART];

// atom type
char atype[10];
//  Function prototypes
//  initialize positions on simple cubic lattice, also calls function to initialize velocities

void initialize();  
//  update positions and velocities using Velocity Verlet algorithm 
//  print particle coordinates to file for rendering via VMD or other animation software
//  return 'instantaneous pressure'

double VelocityVerlet(double dt, int iter, FILE *fp);  

//  Numerical Recipes function for generation gaussian distribution
double gaussdist();

//  Initialize velocities according to user-supplied initial Temperature (Tinit)
void initializeVelocities();

//  Compute mean squared velocity from particle velocities
double MeanSquaredVelocity();

//  Compute total kinetic energy from particle mass and velocities
double Kinetic();

// Compute total potential energy from particle coordinates
double PotentialPrepare(int N, double sigma, double epsilon, double * rx, double * ry, double * rz);

//  Compute Force using F = -dV/dr
//  solve F = ma for use in Velocity Verlet
void computeAccelerationsPrepare(int N);


int main() {
    
    //  variable delcarations
    int i;
    double dt, Vol, Temp, Press, Pavg, Tavg, rho;
    double VolFac, TempFac, PressFac, timefac;
    double KE, PE, mvs, gc, Z;
    char prefix[1000], tfn[1000], ofn[1000], afn[1000];
    FILE *tfp, *ofp, *afp;
    
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  WELCOME TO WILLY P CHEM MD!\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  ENTER A TITLE FOR YOUR CALCULATION!\n");
    scanf("%s",prefix);
    strcpy(tfn,prefix);
    strcat(tfn,"_traj.xyz");
    strcpy(ofn,prefix);
    strcat(ofn,"_output.txt");
    strcpy(afn,prefix);
    strcat(afn,"_average.txt");
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  TITLE ENTERED AS '%s'\n",prefix);
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    
    /*     Table of values for Argon relating natural units to SI units:
     *     These are derived from Lennard-Jones parameters from the article
     *     "Liquid argon: Monte carlo and molecular dynamics calculations"
     *     J.A. Barker , R.A. Fisher & R.O. Watts
     *     Mol. Phys., Vol. 21, 657-673 (1971)
     *
     *     mass:     6.633e-26 kg          = one natural unit of mass for argon, by definition
     *     energy:   1.96183e-21 J      = one natural unit of energy for argon, directly from L-J parameters
     *     length:   3.3605e-10  m         = one natural unit of length for argon, directly from L-J parameters
     *     volume:   3.79499-29 m^3        = one natural unit of volume for argon, by length^3
     *     time:     1.951e-12 s           = one natural unit of time for argon, by length*sqrt(mass/energy)
     ***************************************************************************************/
    
    //  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //  Edit these factors to be computed in terms of basic properties in natural units of
    //  the gas being simulated
    
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("  WHICH NOBLE GAS WOULD YOU LIKE TO SIMULATE? (DEFAULT IS ARGON)\n");
    printf("\n  FOR HELIUM,  TYPE 'He' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR NEON,    TYPE 'Ne' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR ARGON,   TYPE 'Ar' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR KRYPTON, TYPE 'Kr' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR XENON,   TYPE 'Xe' THEN PRESS 'return' TO CONTINUE\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    scanf("%s",atype);
    
    if (strcmp(atype,"He")==0) {
        
        VolFac = 1.8399744000000005e-29;
        PressFac = 8152287.336171632;
        TempFac = 10.864459551225972;
        timefac = 1.7572698825166272e-12;
        
    }
    else if (strcmp(atype,"Ne")==0) {
        
        VolFac = 2.0570823999999997e-29;
        PressFac = 27223022.27659913;
        TempFac = 40.560648991243625;
        timefac = 2.1192341945685407e-12;
        
    }
    else if (strcmp(atype,"Ar")==0) {
        
        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        //strcpy(atype,"Ar");
        
    }
    else if (strcmp(atype,"Kr")==0) {
        
        VolFac = 4.5882712000000004e-29;
        PressFac = 59935428.40275003;
        TempFac = 199.1817584391428;
        timefac = 8.051563913585078e-13;
        
    }
    else if (strcmp(atype,"Xe")==0) {
        
        VolFac = 5.4872e-29;
        PressFac = 70527773.72794868;
        TempFac = 280.30305642163006;
        timefac = 9.018957925790732e-13;
        
    }
    else {
        
        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        strcpy(atype,"Ar");
        
    }
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n                     YOU ARE SIMULATING %s GAS! \n",atype);
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  YOU WILL NOW ENTER A FEW SIMULATION PARAMETERS\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n\n  ENTER THE INTIAL TEMPERATURE OF YOUR GAS IN KELVIN\n");
    scanf("%lf",&Tinit);
    // Make sure temperature is a positive number!
    if (Tinit<0.) {
        printf("\n  !!!!! ABSOLUTE TEMPERATURE MUST BE A POSITIVE NUMBER!  PLEASE TRY AGAIN WITH A POSITIVE TEMPERATURE!!!\n");
        exit(0);
    }
    // Convert initial temperature from kelvin to natural units
    Tinit /= TempFac;
    
    
    printf("\n\n  ENTER THE NUMBER DENSITY IN moles/m^3\n");
    printf("  FOR REFERENCE, NUMBER DENSITY OF AN IDEAL GAS AT STP IS ABOUT 40 moles/m^3\n");
    printf("  NUMBER DENSITY OF LIQUID ARGON AT 1 ATM AND 87 K IS ABOUT 35000 moles/m^3\n");
    
    scanf("%lf",&rho);
    
    N = 5000;
    Vol = N/(rho*NA);
    
    Vol /= VolFac;
    
    //  Limiting N to MAXPART for practical reasons
    if (N>=MAXPART) {
        
        printf("\n\n\n  MAXIMUM NUMBER OF PARTICLES IS %i\n\n  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY \n\n", MAXPART);
        exit(0);
        
    }
    //  Check to see if the volume makes sense - is it too small?
    //  Remember VDW radius of the particles is 1 natural unit of length
    //  and volume = L*L*L, so if V = N*L*L*L = N, then all the particles
    //  will be initialized with an interparticle separation equal to 2xVDW radius
    if (Vol<N) {
        
        printf("\n\n\n  YOUR DENSITY IS VERY HIGH!\n\n");
        printf("  THE NUMBER OF PARTICLES IS %i AND THE AVAILABLE VOLUME IS %f NATURAL UNITS\n",N,Vol);
        printf("  SIMULATIONS WITH DENSITY GREATER THAN 1 PARTCICLE/(1 Natural Unit of Volume) MAY DIVERGE\n");
        printf("  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY AND RETRY\n\n");
        exit(0);
    }
    // Vol = L*L*L;
    // Length of the box in natural units:
    L = pow(Vol,(1./3));
    
    //  Files that we can write different quantities to
    tfp = fopen(tfn,"w");     //  The MD trajectory, coordinates of every particle at each timestep
    ofp = fopen(ofn,"w");     //  Output of other quantities (T, P, gc, etc) at every timestep
    afp = fopen(afn,"w");    //  Average T, P, gc, etc from the simulation
    
    int NumTime;
    if (strcmp(atype,"He")==0) {
        
        // dt in natural units of time s.t. in SI it is 5 f.s. for all other gasses
        dt = 0.2e-14/timefac;
        //  We will run the simulation for NumTime timesteps.
        //  The total time will be NumTime*dt in natural units
        //  And NumTime*dt multiplied by the appropriate conversion factor for time in seconds
        NumTime=50000;
    }
    else {
        dt = 0.5e-14/timefac;
        NumTime=200;
        
    }
    
    //  Put all the atoms in simple crystal lattice and give them random velocities
    //  that corresponds to the initial temperature we have specified
    initialize();
    
    //  Based on their positions, calculate the ininial intermolecular forces
    //  The accellerations of each particle will be defined from the forces and their
    //  mass, and this will allow us to update their positions via Newton's law
    
    computeAccelerationsPrepare(N);
    
    //computeAccelerations();

    // Print number of particles to the trajectory file
    fprintf(tfp,"%i\n",N);
    
    //  We want to calculate the average Temperature and Pressure for the simulation
    //  The variables need to be set to zero initially
    Pavg = 0;
    Tavg = 0;
    
    
    int tenp = floor(NumTime/10);
    fprintf(ofp,"  time (s)              T(t) (K)              P(t) (Pa)           Kinetic En. (n.u.)     Potential En. (n.u.) Total En. (n.u.)\n");
    printf("  PERCENTAGE OF CALCULATION COMPLETE:\n  [");
    for (i=0; i<NumTime+1; i++) {
        
        //  This just prints updates on progress of the calculation for the users convenience
        if (i==tenp) printf(" 10 |");
        else if (i==2*tenp) printf(" 20 |");
        else if (i==3*tenp) printf(" 30 |");
        else if (i==4*tenp) printf(" 40 |");
        else if (i==5*tenp) printf(" 50 |");
        else if (i==6*tenp) printf(" 60 |");
        else if (i==7*tenp) printf(" 70 |");
        else if (i==8*tenp) printf(" 80 |");
        else if (i==9*tenp) printf(" 90 |");
        else if (i==10*tenp) printf(" 100 ]\n");
        fflush(stdout);
        
        
        // This updates the positions and velocities using Newton's Laws
        // Also computes the Pressure as the sum of momentum changes from wall collisions / timestep
        // which is a Kinetic Theory of gasses concept of Pressure
        Press = VelocityVerlet(dt, i+1, tfp);
        Press *= PressFac;
        
        //  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //  Now we would like to calculate somethings about the system:
        //  Instantaneous mean velocity squared, Temperature, Pressure
        //  Potential, and Kinetic Energy
        //  We would also like to use the IGL to try to see if we can extract the gas constant
        mvs = MeanSquaredVelocity();
        KE = Kinetic();
        PE = PotentialPrepare(N, sigma, epsilon, rx, ry, rz);
        //PE = Potential();
        // Temperature from Kinetic Theory
        Temp = m*mvs/(3*kB) * TempFac;
        
        // Instantaneous gas constant and compressibility - not well defined because
        // pressure may be zero in some instances because there will be zero wall collisions,
        // pressure may be very high in some instances because there will be a number of collisions
        gc = NA*Press*(Vol*VolFac)/(N*Temp);
        Z  = Press*(Vol*VolFac)/(N*kBSI*Temp);
        
        Tavg += Temp;
        Pavg += Press;
        
        fprintf(ofp,"  %8.4e  %20.8f  %20.8f %20.8f  %20.8f  %20.8f \n",i*dt*timefac,Temp,Press,KE, PE, KE+PE);
        
        
    }
    
    // Because we have calculated the instantaneous temperature and pressure,
    // we can take the average over the whole simulation here
    Pavg /= NumTime;
    Tavg /= NumTime;
    Z = Pavg*(Vol*VolFac)/(N*kBSI*Tavg);
    gc = NA*Pavg*(Vol*VolFac)/(N*Tavg);
    fprintf(afp,"  Total Time (s)      T (K)               P (Pa)      PV/nT (J/(mol K))         Z           V (m^3)              N\n");
    fprintf(afp," --------------   -----------        ---------------   --------------   ---------------   ------------   -----------\n");
    fprintf(afp,"  %8.4e  %15.5f       %15.5f     %10.5f       %10.5f        %10.5e         %i\n",i*dt*timefac,Tavg,Pavg,gc,Z,Vol*VolFac,N);
    
    printf("\n  TO ANIMATE YOUR SIMULATION, OPEN THE FILE \n  '%s' WITH VMD AFTER THE SIMULATION COMPLETES\n",tfn);
    printf("\n  TO ANALYZE INSTANTANEOUS DATA ABOUT YOUR MOLECULE, OPEN THE FILE \n  '%s' WITH YOUR FAVORITE TEXT EDITOR OR IMPORT THE DATA INTO EXCEL\n",ofn);
    printf("\n  THE FOLLOWING THERMODYNAMIC AVERAGES WILL BE COMPUTED AND WRITTEN TO THE FILE  \n  '%s':\n",afn);
    printf("\n  AVERAGE TEMPERATURE (K):                 %15.10f\n",Tavg);
    printf("\n  AVERAGE PRESSURE  (Pa):                  %15.10f\n",Pavg);
    printf("\n  PV/nT (J * mol^-1 K^-1):                 %15.10f\n",gc);
    printf("\n  PERCENT ERROR of pV/nT AND GAS CONSTANT: %15.10f\n",100*fabs(gc-8.3144598)/8.3144598);
    printf("\n  THE COMPRESSIBILITY (unitless):          %15.10f \n",Z);
    printf("\n  TOTAL VOLUME (m^3):                      %10.10e \n",Vol*VolFac);
    printf("\n  NUMBER OF PARTICLES (unitless):          %i \n", N);
    
    fclose(tfp);
    fclose(ofp);
    fclose(afp);
    
    return 0;
}

double power(double base, int expoent){
    double res = 1.;
    int i = 0;
    while (i < expoent){
        res *= base;
        i++;
    }
    
    return res;

}

void initialize() {
    int n, p, i, j, k;
    double pos;
    
    // Number of atoms in each direction
    n = int(ceil(pow(N, 1.0/3)));
    
    //  spacing between atoms along a given direction
    pos = L / n;
    
    //  index for number of particles assigned positions
    p = 0;
    //  initialize positions
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            for (k=0; k<n; k++) {
                if (p<N) {
                    
                    rx[p] = (i + 0.5)*pos;
                    ry[p] = (j + 0.5)*pos;
                    rz[p] = (k + 0.5)*pos;
                    
                    
                }
                p++;
            }
        }
    }
    
    // Call function to initialize velocities
    initializeVelocities();

    
     /*printf("  Printing initial positions!\n");
     for (i=0; i<N; i++) {
     printf("  %6.3e  %6.3e  %6.3e\n",rx[i],ry[i],rz[i]);
     }
     
     printf("  Printing initial velocities!\n");
     for (i=0; i<N; i++) {
     printf("  %6.3e  %6.3e  %6.3e\n",vx[i],vy[i],vz[i]);
     }*/
    
    
    
    
}   
//  Function to calculate the averaged velocity squared
double MeanSquaredVelocity() { 
    
    double vx2 = 0;
    double vy2 = 0;
    double vz2 = 0;
    double v2;
    
    for (int i=0; i<N; i++) {
        
        vx2 += power(vx[i], 2);
        vy2 += power(vy[i], 2);
        vz2 += power(vz[i], 2);
        
    }
    v2 = (vx2+vy2+vz2)/N;
    
    
   // printf("  Average of x-component of velocity squared is %f\n",v2);
    return v2;
}

//  Function to calculate the kinetic energy of the system
double Kinetic() { //Write Function here!  
    
    double kin;
    
    kin =0.;
    for (int i=0; i<N; i++) {
         
        kin += power(vx[i],2);
        kin += power(vy[i],2);
        kin += power(vz[i],2);
        
    }
    
     //printf("  Total Kinetic Energy is %f\n",kin);
    return (m*kin)/2.;
    
}


//Function to calculate the potential energy of the system

__device__ double power_cuda(double base, int expoent) {
    double res = 1.0;
    int i = 0;
    while (i < expoent) {
        res *= base;
        i++;
    }
    return res;
}

__global__ void Potential_Map(double* drx, double* dry, double* drz, double* Pot, int N, double sigma) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    double r3, r2, r6, pot;
    double sig6 = power_cuda(sigma, 6);
    if (i < N) {
        pot = 0.0;

        for (int j = 0; j < N; j++) {
            if (j != i) {
                r2 = 0.0;

                r2 += power_cuda((drx[i] - drx[j]), 2);
                r2 += power_cuda((dry[i] - dry[j]), 2);
                r2 += power_cuda((drz[i] - drz[j]), 2);

                r3 = power_cuda(r2, 3);
                r6 = power_cuda(r2, 6);

                pot += (sig6 - r3) / r6;
            }
        }
        Pot[i] = pot;
    }
}

__global__ void Potential_Reduce(double *Pot, double *Pot_r, int elem_d_vez) {
    // Allocate shared memory
    __shared__ double partial_sum[256];

    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + (threadIdx.x * elem_d_vez);
    double sum_elem = 0.0; 

   
    
    // Load elements into shared memory
    for (int i = 0; i < elem_d_vez; i++) sum_elem += Pot[i + tid]; 
    
    partial_sum[threadIdx.x] = sum_elem;
     __syncthreads();
    // Increase the stride of the access until we exceed the CTA dimensions
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write its result to main memory
    if (threadIdx.x == 0) {
        Pot_r[blockIdx.x] = partial_sum[0];
    }
}

double PotentialPrepare(int N, double sigma, double epsilon, double * rx, double * ry, double * rz) {
    
    const int TB_SIZE = 256;
    int GRID_SIZE = ceil((N * 1.0) / (TB_SIZE * 1.0));
    
    // M = 5120
    int M = TB_SIZE * GRID_SIZE;
    double Pot[M] = {0};
    
    //double testmap[5120];
    //double testreduce1[5120];

    //for(int i = 0; i < N; i++){ printf("Valor %d é %f, %f, %f \n", i , rx[i], ry[i], rz[i] ); }
    
    double * d_Pot_r;
    double * d_Pot;

    double * drx;
    double * dry;
    double * drz;

    size_t bytes   = N * sizeof(double);
    size_t bytesP   = M * sizeof(double);
    
    cudaMalloc(&d_Pot, bytesP);
    cudaMalloc(&d_Pot_r, bytesP);
    
    cudaMemcpy(d_Pot_r, Pot, bytesP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pot, Pot, bytesP, cudaMemcpyHostToDevice);

    cudaMalloc(&drx, bytes);
    cudaMalloc(&dry, bytes);
    cudaMalloc(&drz, bytes);
    
    cudaMemcpy(drx, rx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dry, ry, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(drz, rz, bytes, cudaMemcpyHostToDevice);

    Potential_Map<<<GRID_SIZE, TB_SIZE>>>(drx, dry, drz, d_Pot, N, sigma);
    //cudaMemcpy(testmap, d_Pot, bytes, cudaMemcpyDeviceToHost);

    // Print the values of testmap
    //for(int i = 0; i < N; i++){ printf("Valor %d é %f \n",i,testmap[i]); }

    Potential_Reduce<<< 1, TB_SIZE>>>(d_Pot, d_Pot_r, GRID_SIZE);
    cudaMemcpy(Pot, d_Pot_r, bytesP, cudaMemcpyDeviceToHost);

    //printf("Valor dado é %f \n", Pot[0]);

    cudaFree(d_Pot);
    cudaFree(d_Pot_r);
    cudaFree(drx);
    cudaFree(dry);
    cudaFree(drz);

    return Pot[0] * 4 * epsilon * power(sigma,6);
}

__device__ double ourAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ double ourAtomicMinus(double* address, double val){
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) - val));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

//   Uses the derivative of the Lennard-Jones potential to calculate
//   the forces on each atom.  Then uses a = F/m to calculate the
//   accelleration of each atom. 
 __global__ void computeAccelerations( double* dax, double* day, double* daz, double* drx, double* dry, double* drz, int N){ 
    int j;
    double f, rSqd;
    double rij0, rij1, rij2;
    double inv3, inv7, inv2;
    int i = threadIdx.x + (blockIdx.x * blockDim.x);

    if( i < N - 1){

        for (j = i + 1; j < N; j++){
            
            // initialize r^2 to zero
            rSqd = 0;
            
            //  component-by-componenent position of i relative to j
            rij0 = (drx[i] - drx[j]);
            rij1 = (dry[i] - dry[j]);
            rij2 = (drz[i] - drz[j]);
            
            //  sum of squares of the components
            rSqd += power_cuda(rij0, 2);
            rSqd += power_cuda(rij1, 2);
            rSqd += power_cuda(rij2, 2);

             //  From derivative of Lennard-Jones with sigma and epsilon set equal to 1 in natural units!
            inv3 =  power_cuda(rSqd, 3);
            inv2 =  power_cuda(inv3, 2) * rSqd;
            inv7 = (2 - inv3)/inv2;
            f    = 24 * inv7;
            
            ourAtomicAdd(&dax[i], rij0 * f);
            ourAtomicAdd(&day[i], rij1 * f);
            ourAtomicAdd(&daz[i], rij2 * f);
            
            ourAtomicMinus( &dax[j], rij0 * f);
            ourAtomicMinus( &day[j], rij1 * f);
            ourAtomicMinus( &daz[j], rij2 * f);
              
        }
    
    }
    
}

void computeAccelerationsPrepare(int N){
    double temp[N] = {0};
    size_t bytes = N * sizeof(double);
    
    double * drx;
    double * dry;
    double * drz;

    double * dax;
    double * day;
    double * daz;
   
    cudaMalloc(&dax, bytes);
    cudaMalloc(&day, bytes);
    cudaMalloc(&daz, bytes);
    cudaMemcpy(dax, temp, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(day, temp, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(daz, temp, bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&drx, bytes);
    cudaMalloc(&dry, bytes);
    cudaMalloc(&drz, bytes);
    cudaMemcpy(drx, rx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dry, ry, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(drz, rz, bytes, cudaMemcpyHostToDevice);
    
    const int TB_SIZE = 256;
    int GRID_SIZE = ceil((N * 1.0) / (TB_SIZE * 1.0));

    computeAccelerations<<<GRID_SIZE, TB_SIZE>>>(dax,day,daz, drx,dry,drz, N);
    
    cudaMemcpy(ax, dax, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(ay, day, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(az, daz, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(drx);
    cudaFree(dry);
    cudaFree(drz);
    cudaFree(dax);
    cudaFree(day);
    cudaFree(daz);

}


// returns sum of dv/dt*m/A (aka Pressure) from elastic collisions with walls
double VelocityVerlet(double dt, int iter, FILE *fp) {
    int i;
    
    double psum = 0.;
    
    //  Compute accelerations from forces at current position
    // this call was removed (commented) for predagogical reasons
    //computeAccelerations();
    //  Update positions and velocity with current velocity and acceleration
    //printf("  Updated Positions!\n");
    for (i=0; i<N; i++) {

        rx[i] += (vx[i] * dt);
        rx[i] += (0.5 * ax[i]  * power(dt,2));

        ry[i] += (vy[i] * dt);
        ry[i] += (0.5 * ay[i] * power(dt,2));
        
        rz[i] += (vz[i] * dt);
        rz[i] += (0.5 * az[i] * power(dt,2));

        vx[i] += (0.5 * ax[i]  * dt);
        vy[i] += (0.5 * ay[i]  * dt);
        vz[i] += (0.5 * az[i]  * dt);
        
        //printf("  %i  %6.4e   %6.4e   %6.4e\n",i,r[i+0],r[i+1],r[i+2]);
    }
    
    //  Update accellerations from updated positions
    
    computeAccelerationsPrepare(N);
    //computeAccelerations();

    //  Update velocity with updated acceleration
    for (i=0; i<N; i++) {

        vx[i] += (0.5 * ax[i] * dt);
        vy[i] += (0.5 * ay[i] * dt);
        vz[i] += (0.5 * az[i] * dt);

    }
    
    // Elastic walls
for (i = 0; i < N; i++) {
    if (rx[i] < 0 || rx[i] >= L) {
        vx[i] *= -1.0;  // Elastic walls
        psum += fabs(vx[i]);  // Contribution to pressure from walls
    }

    if (ry[i] < 0 || ry[i] >= L) {
        vy[i] *= -1.0;  // Elastic walls
        psum += fabs(vy[i]);  // Contribution to pressure from walls
    }

    if (rz[i] < 0 || rz[i] >= L) {
        vz[i] *= -1.0;  // Elastic walls
        psum += fabs(vz[i]);  // Contribution to pressure from walls
    }
}
        
/*
for (i=0; i<N; i++) {
    fprintf(fp,"%s",atype);
    for (j=0; j<3; j++) {
        fprintf(fp,"  %12.10e ",r[i+j]);
    }
    fprintf(fp,"\n");
}
fprintf(fp,"\n \n");
*/
    
    return (psum * m)/(3*L*L*dt);
}
void initializeVelocities() {
    
    int i;
    
    for (i=0; i<N; i++) {
        

        vx[i] = gaussdist();
        vy[i] = gaussdist();
        vz[i] = gaussdist();
        
    }
    
    // Vcm = sum_i^N  m*v_i/  sum_i^N  M
    // Compute center-of-mas velocity according to the formula above
    double vCM[3] = {0, 0, 0};
    
    for (i=0; i<N; i++) {
        
        vCM[0] += m * vx[i];
        vCM[1] += m * vy[i];
        vCM[2] += m * vz[i];

    }
    
    
    for (i=0; i<3; i++) vCM[i] /= N*m;
    
    //  Subtract out the center-of-mass velocity from the
    //  velocity of each particle... effectively set the
    //  center of mass velocity to zero so that the system does
    //  not drift in space!
    for (i=0; i<N; i++) {
        
        vx[i] -= vCM[0];
        vy[i] -= vCM[1];
        vz[i] -= vCM[2];
    }
    
    //  Now we want to scale the average velocity of the system
    //  by a factor which is consistent with our initial temperature, Tinit
    double vSqdSum, lambda;
    vSqdSum=0.;
    for (i=0; i<N; i++) {

        vSqdSum += power(vx[i],2);
        vSqdSum += power(vy[i],2); 
        vSqdSum += power(vz[i],2); 
        
    }
    
    lambda = sqrt( 3*(N-1)*Tinit/vSqdSum);
    
    for (i=0; i<N; i++) {
        
        vx[i] *= lambda;
        vy[i] *= lambda;
        vz[i] *= lambda;
    }
    
    /*
    for (i=0; i<N3; i+=3) {
        printf(" %f\n %f\n %f\n",vx[i],vy[i],vz[i]);
    }
    */

}


//  Numerical recipes Gaussian distribution number generator
double gaussdist() {
    static bool available = false;
    static double gset;
    double fac, rsq, v1, v2;
    if (!available) {
        do {
            v1 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            v2 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        available = true;
        
        return v2*fac;
    } else {
        
        available = false;
        return gset;
        
    }
}