#ifndef MYTOOLS_HPP
#define MYTOOLS_HPP

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;


void SMSumRows(const SparseMatrix &mat, Vector &vec);

std::string findWordN(const std::string& filename, const std::string& targetWord, int N);
int HowManyWire(const std::string& filename);
double IntScalar3(FiniteElementSpace &fes, Coefficient &coeff, int Attr);
double IntScalar4(FiniteElementSpace &fes, Coefficient &coeff, int Attr);
/// Function calculating the integral of a scalar Coefficient, Coeff, over a domain
/// identified by an Attribute, Attr, on the FiniteElementSpace fes.
double IntegrateScalar(FiniteElementSpace &fes, Coefficient &coeff, int Attr);
void Glvis(Mesh *m, GridFunction *gf, string title, int precision = 8, string keys=" keys 'mmcj'");
int GetScalarValues(Array<double> &ap,  GridFunction &gf, Array<double> &av, int vdim);

#endif // MYTOOLS_HPP