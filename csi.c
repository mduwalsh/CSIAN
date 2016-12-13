#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
 
#define DEBUG 1
#if DEBUG
#define __USE_GNU
#include <fenv.h>       /* enable floating exceptions */
unsigned long Where;    // debugging counter
#endif

#include "rand.c"

#define UsVsNature_Them 1    // 1: Us vs Nature game; 2: Us vs Them game
#define PIB_MFA 1            // 1: Predicting individual behavior; 2: Mean-field approximation // forecast method

#define INIT_COM_EFFORT rnd(2)              // 0 or 1
#define INIT_LEAD_PUN_EFFORT 0.5
#define INIT_LEAD_NORM_EFFORT 0.1

// change values to 0 if update strategy for any one of commoner or lead is to be turned off
#define UPDATE_COM 1
#define UPDATE_LEAD_PUN_EFFORT 1
#define UPDATE_LEAD_NORM_EFFORT 1

#define CLUSTER 0            // is this simulation on cluster yes or no (1 or 0)
#define SKIP 10           // time interval between snapshot of states
#define STU 200             // summary time period for simulation

#define AVERAGE_GRAPH_ONLY 0  // if 1, generate only average graphs and supress individual run graphs
#define ALLDATAFILE 0        // if 1, generate all data files for individual runs and summary too 
#define GRAPHS      0        // if 1, saves graphs as png files 


void *Malloc(size_t size)
{
  void *p = malloc(size);

  if (p) return p;
  printf("malloc failed\n");
  _exit(2);
}

#define malloc(x) Malloc(x) 

void *Calloc(size_t nmemb, size_t size)
{
  void *p = calloc(nmemb, size);

  if (p) return p;
  printf("calloc failed\n");
  _exit(3);
}

#define calloc(x,y) Calloc(x,y) 

#define MAX(x,y)         (((x)>(y))? (x): (y))
#define MIN(x,y)         (((x)<(y))? (x): (y))

// system structure
typedef struct
{
  unsigned int  x;            // 1 or 0 
  double        pi;           // payoff   
  double        uc;           // utility function
} Commoner;

typedef struct
{
  double        y;            // punishment effort
  double        z;            // norm promoting effort
  double        pi;           // payoff 
  double        ul;           // utility function
} Leader;


typedef struct
{
  double        P;            // group production P(x) value
  double        X;            // sum of efforts from commoners in group
  double        theta;        // theta: tax on commoners in a group 
  
  unsigned int  nc;           // number of commoners in a group
  Leader        *lead;        // leader
  Commoner      *com;         // group members / commoners; 
} group;

typedef struct
{
  unsigned int ng;            // no. of groups in a polity  
  double SX;                  // sum of group efforts
  group        *g;            // polity members 
} polity;

polity *Polity;               // "whole system"

// system structure ends
// Global variables
/** configuration parameters **/
unsigned Seed;                    // Seed: Seed of psuedorandom number from config file
unsigned long Seed_i;             // Seed_i: seed of each run
unsigned int Runs;                         // no. of runs
unsigned int G;                            // no. of groups
unsigned int n;                            // no. of commoners
double b;                                  // benefit factor
unsigned int T;                            // time period of simulation
double Theta;                     // taxes / rewards
double cx, cy, cz;                // cost parameter
double Sigma;                     // standard deviation for distribution of mutation for contribution
double V1, V2, V3;                // probability with which participants choose method of changing efforts 
double x_0, X0;            // x_0: half effort of commoner; X0: half group effort
double k, delta;        // punishments parameter
double m;                   // probability of migration
double s0, s1;
double Lambda;
unsigned int K;                   // candidate strategies

// statistical variables
double *xmean;          // average effort by individuals
double *ymean;            // average effort by leaders
double *zmean;            // average effort by chiefs
double *pi0mean;                  // average payoff of commoners
double *pi1mean;                  // average payoff of leaders
double *Pmean;                    // group production
double *ucmean;                   // commoners utility function
double *ulmean;                   // leader utility function

dist *Vdist;                      // 'dist' is defined in rand.c; Probability distribution of choosing method of changing efforts

#define EXPECT(a,b,c) if ((a) != fscanf(f, b"%*[ ^\n]\n",c)){ fclose(f); printf("Error: %s\n",b); return 1; }

int read_config(char *file_name)
{
  FILE *f;

  if (!(f = fopen(file_name,"r"))) return 1;
  EXPECT(1, "unsigned Seed      = %u;", &Seed);
  EXPECT(1, "int      Runs      = %d;", &Runs);
  EXPECT(1, "int      T         = %d;", &T);
  
  EXPECT(1, "int      n         = %d;", &n);
  EXPECT(1, "int      G         = %d;", &G); 
  EXPECT(1, "int      K         = %d;", &K);
  
  EXPECT(1, "double   b         = %lf;", &b);  
  EXPECT(1, "double   cx        = %lf;", &cx);
  EXPECT(1, "double   cy        = %lf;", &cy);
  EXPECT(1, "double   cz        = %lf;", &cz);     
  
  EXPECT(1, "double   k         = %lf;", &k);
  EXPECT(1, "double   delta     = %lf;", &delta);  
  EXPECT(1, "double   Theta     = %lf;", &Theta);
 
  EXPECT(1, "double   V1        = %lf;", &V1);
  EXPECT(1, "double   V2        = %lf;", &V2);
  EXPECT(1, "double   V3        = %lf;", &V3); 
  EXPECT(1, "double   m         = %lf;", &m);
  
  EXPECT(1, "double   x0        = %lf;", &x_0);  
  EXPECT(1, "double   X0        = %lf;", &X0);  
  
  EXPECT(1, "double   s0        = %lf;", &s0);  
  EXPECT(1, "double   s1        = %lf;", &s1); 
  EXPECT(1, "double   Lambda    = %lf;", &Lambda);    
  
  EXPECT(1, "double   Sigma     = %lf;", &Sigma);

  fclose(f);   
  
  if (Runs < 1) exit(1);
  return 0;
}
#undef EXPECT

void prep_file(char *fname, char *apndStr)
/*
 * fname: variable in which name of file to be stored
 * apndStr: string to be appended to file name string
 */
{
  //sprintf(fname, "g%02dn%02db%0.2fB%.2ftu%.2ftd%.2feu%.2fed%.2fcx%.2fcy%.2fcz%.2fv1%.2fv2%.2fv3%.2fY0%.2fe%.2fE%.2fx0%.2fy0%.2fz0%.2f%s", G, n, b, B, Theta_u, Theta_d, Eta_u, Eta_d, cx, cy, cz, V1, V2, V3, Y0, e, E, x_0, y_0, z_0, apndStr);
}

// generates random number following exponential distribution
double randexp(double lambda)
/*
 * lambda: rate parameter of exponential distribution
 */
{
  return -log(1.0-U01()) / lambda;  
}

// merge sort
void merge (double *a, int n, int m) {
    int i, j, k;
    double *x = malloc(n * sizeof(double));
    for (i = 0, j = m, k = 0; k < n; k++) {
        x[k] = j == n      ? a[i++]
             : i == m      ? a[j++]
             : a[j] < a[i] ? a[j++]
             :               a[i++];
    }
    for (i = n; i--;) {
        a[i] = x[i];
    }
    free(x);
}
 
void merge_sort (double *a, int n) {
    if (n < 2)
        return;
    int m = n>>1;   // divide by 2
    merge_sort(a, m);
    merge_sort(a + m, n - m);
    merge(a, n, m);
}

void merge_key_value (int *a, double *v,  int n, int m) {
    int i, j, k;
    int *x = malloc(n * sizeof(int));
    for (i = 0, j = m, k = 0; k < n; k++) {
        x[k] = j == n      ? a[i++]
             : i == m      ? a[j++]
             : v[a[j]] < v[a[i]] ? a[j++]
             :               a[i++];
    }
    for (i = n; i--;) {
        a[i] = x[i];
    }
    free(x);
}

void merge_sort_key_value (int *a, double *v, int n) {
    if (n < 2)
        return;
    int m = n>>1;   // divide by 2
    merge_sort_key_value(a, v, m);
    merge_sort_key_value(a + m, v, n - m);
    merge_key_value(a, v, n, m);
} 

// allocate memory for system
void setup()
{
  int j;
  group *g;
  // allocate memory for Polity and groups and commoners
  Polity = malloc(sizeof(polity));                                // one polity 
  Polity->g = malloc(G*sizeof(group));                            // allocate memory for groups in each polity    
  for(j = 0; j < G; j++){                                         // through all groups in a polity
    g = Polity->g+j;                                              // get reference to group pointer
    g->lead = malloc(sizeof(Leader));                             // allocate memory for leader in a group
    g->com = malloc(n*sizeof(Commoner));                          // allocate memory for commoners in a group
  }
  Vdist = allocdist(4);                                           // allocate memory for strategy update method probability distribution 
}

void cleanup()                                                       // cleans Polity system memory
{
  int j;
  group *g;  
  for(j = 0; j < G; j++){                                          // through all groups in a polity
    g = Polity->g+j;                                               // get reference to group pointer
    free(g->lead); 
    free(g->com);
  }
  free(Polity->g);  
  free(Polity);
  freedist(Vdist);
}

/*
void allocStatVar()
{
  // allocate memory to store statistics points of snapshot along time period of simulaltion
  xmean = calloc((int)(T/SKIP+1), sizeof(double));         // for effort by commoners
  ymean = calloc((int)(T/SKIP+1), sizeof(double));         // for effort by leaders
  zmean = calloc((int)(T/SKIP+1), sizeof(double));         // for effort by chiefs
  pi0mean = calloc((int)(T/SKIP+1), sizeof(double));                   // for payoff of commoners
  pi1mean = calloc((int)(T/SKIP+1), sizeof(double));                   // for payoff of leaders
  pi2mean = calloc((int)(T/SKIP+1), sizeof(double));                   // for payoff of chiefs
  pmean = calloc((int)(T/SKIP+1), sizeof(double));                     // for punishment effort by leaders 
  qmean = calloc((int)(T/SKIP+1), sizeof(double));                     // for punishment effort by chiefs
  TUmean = calloc((int)(T/SKIP+1), sizeof(double));                    // for tax paid by commoners to leaders
  EUmean = calloc((int)(T/SKIP+1), sizeof(double));                    // for tax paid by leaders to chiefs
  TDmean = calloc((int)(T/SKIP+1), sizeof(double));                    // for tax paid by commoners to leaders
  EDmean = calloc((int)(T/SKIP+1), sizeof(double));                    // for tax paid by leaders to chiefs
  Pmean = calloc((int)(T/SKIP+1), sizeof(double));                    // for group production 
  Qmean = calloc((int)(T/SKIP+1), sizeof(double));                    // for polity production // singling out B from BQ  
  Mdist = allocdist(4);                                              // allocate memory for distribution for update strategy method
}



void clearStatVar()
{
  free(xmean); free(ymean); free(zmean); free(pi0mean); free(pi1mean); free(pi2mean); free(pmean); free(qmean);
  free(TUmean); free(EUmean); free(TDmean); free(EDmean);
  free(Pmean); free(Qmean);
  freedist(Mdist);  
}
*/

double P(double X, double SX)
/*
 * X : group effort
 * SX: sum of group efforts in whole group
 */
{    
#if UsVsNature_Them == 1  // us vs nature 
  return X/( X + n*x_0 );
#endif
#if UsVsNature_Them == 2  // us vs them
  return (X*G) / SX;
#endif
}

// returns frequency of contributors in group
double q( double X)
{
  return X/n;
}

// return strength of norm internalization
double s(double X)
{
  return s0 + s1*q(X);  
}

// utility function
double u(unsigned int x1, unsigned int x0, double X, double y, double z, double SX)
{  
  double _s = s(X);
  double pi_c = (1.0-Theta)*b*P(X-x0+x1,SX-x0+x1) - cx*x1 - k*y*(1.0-x1);            // expected material payoff for commoners
  return (1.0-_s)*pi_c + _s*z*x1;
}

// returns x' using myopic optimization for commoners
double f(int x, double X, double y, double z, double SX)
{
  double u0, u1;
  u0 = u(0, x, X, y, z, SX);                               // utility function when x' = 0
  u1 = u(1, x, X, y, z, SX);                               // utility function when x' = 1
  return ( U01() < 1.0/(1.0+exp(Lambda*(u0-u1))) )? 1: 0;  
}

#if PIB_MFA == 1            // Predicting individual behavior
// returns new forecasted sum of efforts from commoners given y and z
double F(double X, double y, double z, unsigned int *x0, unsigned int *x1, double SX)
/*
 * X: sum of efforts from commoners in group
 * y: punishment effort
 * z: norm effort
 * x0: efforts array of commoners
 * x1: new forecasted efforts array of commoners
 */
{
  int i;
  double s;
  for(s = 0, i = 0; i < n; i++){
    x1[i] = f(x0[i], X, y, z, SX);                // new forecasted efforts of commoners
    s += x1[i];
  }
  return s;
}
#endif
#if PIB_MFA == 2         // Mean-field approximation
// returns new forecasted sum of efforts from commoners given y and z
double F(double X, double y, double z)
{
  double s = s(X);
  double B = (1.0 - Theta)*b;
  double C = cx - k*y - (s*z)/(1.0-s);
#if UsVsNature_Them == 1  
  double R = B/(C*X0);
  if(R > 1.0)
    return X0*(sqrt(R) - 1.0);
  else
    return 0;
#endif
#if UsVsNature_Them == 2
  return B/C;
#endif
}
#endif

// returns utility function leader trying to maximize
double u_l(double y, double z, double X1, double X2, double SX2)
{
  return -cy*n*y - cz*z - delta*(n-X1)*y + Theta*b*P(X2, SX2);
}

// calculates group effort X
void calcX(int j)
{
  int i;  
  double sx;
  group *g = Polity->g+j;
  for(sx = 0, i = 0; i < n; i++){                       // through all commoners
    sx += g->com[i].x;                                   // sum production efforts from commoners
  }
  g->X = sx;                                                // sum of efforts from commoners  
}

// calculates group production P
void calcP(int j, double SX)
{
  group *g = Polity->g+j;
  g->P = P(g->X, SX);
}

// calculates payoff of a group j
void calcPayoff(int j)
{
  int i;   
  group *g;
  Leader *ld;
  Commoner *com;
  double gpc, cp;    
  
  g = Polity->g+j;
  ld = g->lead;
  gpc = (1.0 - g->theta)*b*g->P;                       // group production share to each commoner 
  // update payoff of commoners 
  for(cp = 0, i = 0; i < n; i++){
    com = g->com+i;
    com->pi = gpc - cx*com->x;                               // payoff due to group production - cost of x 
    if(com->x == 0){
      if(U01() < ld->y){
	com->pi -= k;                                        // payoff after punishment from leader with probability y
	cp += delta;
      }
    }
    
  }
  // update payoff of leader
  ld->pi = g->theta*n*b*g->P -cy*n*ld->y -cz*ld->z -cp; // payoff due to group production - cost of y - cost of punishing   
}

void calcUtilityFunction(int j, double SX)
{
  int i;
  Commoner *com;
  Leader *ld;  
  group *g;
  g = Polity->g+j;
  ld = g->lead;
  for(i = 0; i < n; i++){
    com = g->com+i;
    com->uc = u(com->x, com->x, g->X, ld->y, ld->z, SX);
  }
  ld->ul = u_l(ld->y, ld->z, g->X, g->X, SX);
}

// updates strategy of commoner
void updateStrategyCommoner(int v, int j, int i, double y, double z, double SX)
/*
 * v: type of strategy update
 * j: group index
 * i: commoner index
 * y: leader's punishing effort
 * z: leader's norm promoting effort
 * SX: sum of group efforts across all groups
 */
{
  group *g;
  Commoner *com;
  g = Polity->g+j;
  com = g->com+i;
  // random mutation
  if(v == 1){
    com->x = rnd(2);  // 0 or 1    
  }
  // selective copying
  else if(v == 2){
    int a;
    if(U01() < (1.0-m)){                                       // copy from same group
      if(n > 1){
	do{ a = rnd(n); }while( a == i);                        // select another individual in group
	if(g->com[a].pi > com->pi){                             // if selected individual has higher payoff, copy strategies
	  com->x = g->com[a].x;
	}
      }
    }
    else{                                                     // choose commoner from another group c in same polity
      int c = j;
      if(G > 1){                                              // if more than one group in polity, choose different group
	do{ c = rnd(G); }while(c == j);	                  // select group c (another group to copy from)
      }      
      a = rnd(n);                                             // select another individual in group b
      if(Polity->g[c].com[a].pi > com->pi){
	com->x = Polity->g[c].com[a].x;
      }
    }    
  }
  // myopic optimization
  else if(v == 3){    
    com->x = f(com->x, g->X, y, z, SX);  
  }
}

// updates strategy of leader // random mutation
void updateStrategyLeader_V1(int j)
{
  group *g;
  Leader *ld;
  g = Polity->g+j;
  ld = g->lead;    
#if UPDATE_LEAD_PUN_EFFORT
    ld->y = MIN(MAX(normal(ld->y, Sigma), 0.0), 1.0);
#endif
#if UPDATE_LEAD_NORM_EFFORT
    ld->z = MIN(MAX(normal(ld->z, Sigma), 0.0), 1.0);
#endif
}

// updates strategy of leader // selective copying
void updateStrategyLeader_V2(int j)
{
  group *g;
  Leader *ld;
  g = Polity->g+j;
  ld = g->lead;
  int a = j;
  if(G > 1){
    do{ a = rnd(G); }while( a == j);                                   // select another group in polity
    if(Polity->g[a].lead[0].pi > ld->pi){                              // if leader in selected group has higher payoff copy strategy
#if UPDATE_LEAD_PUN_EFFORT
      ld->y = (Polity->g+a)->lead->y;                          // copy punishment effort	
#endif
#if UPDATE_LEAD_NORM_EFFORT
      ld->z = (Polity->g+a)->lead->z;                          // copy norm effort
#endif
    }
  }
}

// updates strategy of leader // myopic optimization
// Note: this method changes value of x0 array 
void updateStrategyLeader_V3(int j, unsigned int *x0, double X, double SX)
/*
 * j: index of group of leader
 * x0: array of unchanged this round efforts from commoners in group j
 * X: group effort of this round
 * SX: sum of group efforts across all groups
 */
{
  group *g;
  Leader *ld;
  g = Polity->g+j;
  ld = g->lead;  
  
  int i;
  double X1, SX1, s;
#if PIB_MFA == 1
  unsigned int *x1 = malloc(n*sizeof(unsigned int));             // new forecasted efforts by commoners which sum up to X1 group effort
#endif
  double *X2 = malloc((K+1)*sizeof(double));                    // forecasted Xs for (K+1) candidate y and z 
  double *y = malloc((K+1)*sizeof(double));
  double *z = malloc((K+1)*sizeof(double));
  double *ul = malloc((K+1)*sizeof(double));                    // utility function array    
  dist *ul_dist = allocdist(K+1);                               // utility function distribution array  
  
  // candidate strategies
  y[0] = ld->y;
  z[0] = ld->z;  
  for(i = 1; i < K+1; i++){
#if UPDATE_LEAD_PUN_EFFORT
    y[i] = normal(ld->y, Sigma);
#else
    y[i] = ld->y;
#endif
#if UPDATE_LEAD_NORM_EFFORT
    z[i] = normal(ld->z, Sigma);
#else
    z[i] = ld->z;
#endif
  }        
  
#if PIB_MFA == 1      // Predicting individual behavior
  X1 = F(X, ld->y, ld->z, x0, x1, SX);                      // forecasted X for present y and z
#else                 // Mean-field approximation
  X1 = F(X, ld->y, ld->z);
#endif
  
  SX1 = SX - X + X1;                                        // new sum of group efforts across all groups
  
  for(s = 0, i = 0; i < K+1; i++){      
#if PIB_MFA == 1
    X2[i] = F(X1, y[i], z[i], x1, x0, SX1);                 // next round forecasted group effort by commoners
#else
    X2[i] = F(X1, y[i], z[i]);
#endif
    ul[i] = u_l(y[i], z[i], X1, X2, SX1-X1+X2[i]);                    // utility function
    ul_dist->p[i] = exp(ul[i]*Lambda);
    s += ul_dist->p[i];
  }
  
  initdist(ul_dist, s);                                      // initialize distribution
  // update y and z
  i = drand(ul_dist);
  ld->y = y[i];
  ld->z = z[i];
  
#if PIB_MFA == 1
  free(x1); 
#endif
  free(X2);
  free(y); free(z);
  free(ul); freedist(ul_dist);  
}

void updateStrategy(int j, double SX)
{
  group *g = Polity->g+j;
  Leader *ld = g->lead;
  int i, v;
  double y, z;
  y = ld->y;
  z = ld->z;
#if UPDATE_LEAD_PUN_EFFORT || UPDATE_LEAD_NORM_EFFORT
  v = drand(Vdist);
  if(v == 1){
    updateStrategyLeader_V1(j);
  }
  else if(v == 2){
    updateStrategyLeader_V2(j);
  }
  else if(v == 3){
    unsigned int *x0 = malloc(n*sizeof(unsigned int));
    for(i = 0; i < n; i++){
      x0[i] = (g->com+i)->x;
    }
    updateStrategyLeader_V3(j, x0, g->X, SX);        // note: elements' value of x0 changes
    free(x0);
  }
#endif
  // update commoners
#if UPDATE_COM
  for(i = 0; i < n; i++){
    v = drand(Vdist);
    if(v){
      updateStrategyCommoner(v, j, i, y, z, SX);
    }
  }
#endif
}

// play game
void playGame()
{
  int j;
  double SX;
  group *g;
  // update X for every group in this round
  for(SX = 0, j = 0; j < G; j++){
    g = Polity->g+j;
    calcX(j);                       // calculates and assigns X 
    SX += g->X;
  }
  Polity->SX = SX;                  // sum of group efforts for this round
  // update group production for every group in this round
  for(j = 0; j < G; j++){
    calcP(j, SX);                   // updates group production value for each group
    calcPayoff(j);                  // updates payoff of groups
    updateStrategy(j, SX);             // updates strategy for group j
  }
}

// initialize variables
void init()
{
  group *g;
  Leader *ld;
  Commoner *com;
  int i, j;
  for(j = 0; j < G; j++){
    g = Polity->g+j;
    ld = g->lead;
    for(i = 0; i < n; i++){
      com = g->com+i;
      com->x = INIT_COM_EFFORT;
      com->pi = 0;
      com->uc = 0;
    }
    ld->y = INIT_LEAD_PUN_EFFORT;
    ld->z = INIT_LEAD_NORM_EFFORT;
    ld->pi = 0;
    ld->ul = 0;
    g->nc = n;
    g->theta = Theta;
    g->X = 0;
    g->P = 0;
  }
  Polity->SX = 0;
  Polity->ng = G;
  // initialize strategy update method probability distribution
  Vdist->p[0] = 1-V1-V2-V3;
  Vdist->p[1] = V1; 
  Vdist->p[2] = V2;
  Vdist->p[3] = V3;
  initdist(Vdist, 1);      
}

int main(int argc, char **argv)
{
#if DEBUG
  feenableexcept(FE_DIVBYZERO| FE_INVALID|FE_OVERFLOW); // enable exceptions
#endif
  if(argc ^ 2){
    printf("Usage: ./csi csi.config\n");
    exit(1);
  }
  if(read_config(argv[1])){           // read config
    printf("READDATA: Can't process %s \n", argv[1]);
    return 1;
  }    
  
  initrand(Seed);
  //allocStatVar();                                             // allocate memory for global statistic variables
  int r, i;
  unsigned long seed;  
#if !ALLDATAFILE
  char xdata[200], str[30];
#endif
  time_t now;     
  // print headers for values to be displayed on std output
#if !CLUSTER
  printf("\nValues:\t  x\t  y\t  z\t  pi_0\t  pi_1\t   uc\t  ul\t  P\t  seed\n");
#endif
  
  for( r = 0; r < Runs; r++){                                 // through all sets of runs    
    // initialize pseudorandom generator with seed
    if(Seed == 0){      
      now = time(0);
      seed = ((unsigned long)now) + r;  
    }
    else{
      seed = Seed;
    }
    //seed = 1461867884;
    Seed_i = seed;
    initrand(seed);      
    
    //printf("\n run# %d seed: %lu\n",r+1, seed);
    // setup and initialize variables
    setup();                                                  // allocates memory for all polity system variables
    init();                                                   // intialize polity system state values
    //calcStat(0, r);
    for( i = 0; i < T; i++){                                  // through all time points of simulation
      playGame();                                             // play us vs nature and us vs them game and update strategies
      if( (i+1) % SKIP == 0){                                 // every SKIP time, take snapshot of states of traits	
	//calcStat((i+1)/SKIP, r);                              // calculate statistics and write individual runs data to file
      }   
    }
    //calcStat(-1, -1);                                        // free file pointers for individual run data files
    /*
#if !CLUSTER
    
#if !AVERAGE_GRAPH_ONLY
    if(Runs > 1)
      plotallIndividualRun(r, 0); 
#endif
    
#if GRAPHS
    if(Runs > 1)
      plotallIndividualRun(r, 1);
#endif
    // remove individual datafiles if ALLDATAFILE set to 0
#if !ALLDATAFILE        
      sprintf(str, "x%d.dat", r); prep_file(xdata, str); remove(xdata);
      sprintf(str, "p%d.dat", r); prep_file(xdata, str); remove(xdata);
      sprintf(str, "u%d.dat", r); prep_file(xdata, str); remove(xdata);
      sprintf(str, "gp%d.dat", r); prep_file(xdata, str); remove(xdata);           
#endif
      
#endif
      */
    cleanup();                                                // free all memory allocated for polity system
  }
  /*writeDataToFile();  
  clearStatVar();                                             // free other memory allocated for statistics variables
#if !CLUSTER
  plotall(0);
#if GRAPHS
  plotall(1);
#endif
#endif    */
  return 1;
}

/* 
gcc -Wall -O2 -march=native -pipe -o csi csi.c -lm
./csi csi.config
valgrind -v --track-origins=yes --leak-check=full --show-leak-kinds=all ./csi csi.config
gcc -g -o csi csi.c -lm
gdb csi
run csi.config

//profiling code
gcc -Wall -O2 -march=native -pipe -pg csi.c -o csi -lm
./csi csi.config
 gprof csi gmon.out > analysis.txt  
*/




