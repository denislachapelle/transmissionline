#include "mfem.hpp"
#include "mytools.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace mfem;




void CleanOutDir()
{
    system("rm -f out/*");
}
 

void SMSumColumns(const SparseMatrix &mat, Vector &vec)
{
   int num_rows = mat.NumRows();
   for(int i = 0; i<num_rows; i++) 
   {
      // Get the data for the specified row
      const int *columns = mat.GetRowColumns(i);
      const double *values = mat.GetRowEntries(i);
      int row_size = mat.RowSize(i);
      // Search for the column index
      for (int k = 0; k < row_size; k++)
      {
         vec(columns[k]) += mat.Elem(i, k);
      }
   }
   return;    
}


double ComputeElementArea(FiniteElementSpace *fes, int element_index)
{
    ElementTransformation *trans = fes->GetMesh()->GetElementTransformation(element_index);
    const IntegrationRule &ir = IntRules.Get(fes->GetMesh()->GetElementBaseGeometry(element_index),
                                             fes->GetElementOrder(element_index) * 2);

    double area = 0.0;
    for (int i = 0; i < ir.GetNPoints(); i++) {
        const IntegrationPoint &ip = ir.IntPoint(i);
        trans->SetIntPoint(&ip);
        double detJ = trans->Jacobian().Det();
        area += ip.weight * detJ;
    }
    return area;
}

// Custom Coefficient Class
class ElementCoefficient : public Coefficient
{
private:
    int target_element; // The ID of the target element

public:
    // Constructor
    ElementCoefficient(int elem_id) : target_element(elem_id) {}

    // Override Eval to return 1.0 in the target element and 0.0 elsewhere
    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
    {
        return (T.ElementNo == target_element) ? 1.0 : 0.0;
    }
};

std::string findWordN(const std::string& filename, const std::string& targetWord, int N)
{
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Unable to open file\n";
        return "";
    }
   std::istringstream iss(line);
   std::string firstWord, secondWord, thirdWord, word;

    while (std::getline(file, line))
    {
       iss.str(line);
       iss.clear();
        // Read the first and second words from the line
        if(N==2)
        {
           if (iss >> firstWord >> secondWord)
           {
              if (firstWord == targetWord)
              {
                 word = secondWord;  // Return the second word if the first matches
              }
           }
        }
        else if(N==3)
        {
           if (iss >> firstWord >> secondWord >> thirdWord)
           {
              if (firstWord == targetWord)
              {
                 word = thirdWord;  // Return the second word if the first matches
              }
           }
        }
        else word = "";
    }
    file.close();
    return word;  // Return an empty string if no match is found
}


int HowManyWire(const std::string& filename)
{
   int nbrWire = -1;
   string wirexshape;
   bool loop = true;
   while(loop)    
   {
      nbrWire++;
      char buffer[100];
      snprintf(buffer, sizeof(buffer), "wire%dshape", nbrWire+1);
      wirexshape = buffer;
      if(findWordN(filename, wirexshape, 2) == "") loop = false;
   }
   return nbrWire;
}

double IntScalar3(FiniteElementSpace &fes, Coefficient &coeff, int Attr)
{
   QuadratureSpace qs(fes.GetMesh(), 2*fes.GetMaxElementOrder());
   Array<int> attrs;
   if (fes.GetMesh()->attributes.Size())
   {
      attrs.SetSize(fes.GetMesh()->attributes.Max());
      attrs=0;
      attrs[Attr-1] = 1;
   }
   RestrictedCoefficient restr_coeff(coeff, attrs);
   return qs.Integrate(restr_coeff);
}

double IntScalar4(FiniteElementSpace &fes, Coefficient &coeff, int Attr)
{
   QuadratureSpace qs(fes.GetMesh(), 2*fes.GetMaxElementOrder());
   Array<int> attrs(2); attrs=0;
   assert(fes.GetMesh()->attributes.Max() >= Attr);
   attrs[0] = Attr;
   RestrictedCoefficient restr_coeff(coeff, attrs);
   return qs.Integrate(restr_coeff);
}

/// Function calculating the integral of a scalar Coefficient, Coeff, over a domain
/// identified by an Attribute, Attr, on the FiniteElementSpace fes.
double IntegrateScalar(FiniteElementSpace &fes, Coefficient &coeff, int Attr)
{
   double integral_value = 0.0;
   for (int i = 0; i < fes.GetMesh()->GetNE(); i++)  // Loop over elements
   {
      if(fes.GetAttribute(i) == Attr)
      {   
         ElementTransformation *trans = fes.GetMesh()->GetElementTransformation(i);
         const FiniteElement &fe = *(fes.GetFE(i));
         // Use a quadrature rule that matches the element's order
         const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder());
         for (int j = 0; j < ir.GetNPoints(); j++)  // Loop over quadrature points
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            trans->SetIntPoint(&ip);
            // Evaluate scalar function at the quadrature point in physical coordinates
            double scalar_value = coeff.Eval(*trans, ip);
            // Accumulate the integral (scalar value * weight * Jacobian determinant)
            integral_value += scalar_value * ip.weight * trans->Weight();
         }
      }
   }
   return integral_value;
}

void Glvis(Mesh *m, GridFunction *gf, string title, int precision, string keys)
{
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(precision);
   
   sol_sock << "solution\n" << *m << *gf
            << "window_title '" + title +"'"
            << keys << flush;  

   // Save in GLVIS.
   char fileName[250];
   sprintf(fileName, "glvis/%s.gf", title.data());
   gf->Save(fileName);
}


int GetScalarValues(Array<double> &ap,  GridFunction &gf, Array<double> &av, int vdim)
{
    // DL24112: inspired from intgrad.
    //Transfert the points in the matrix.
    assert( vdim == 1 || vdim == 2 || vdim == 3);
    assert(ap.Size()%vdim==0);
   int NbrPoints = ap.Size()/vdim;
   
   DenseMatrix point_mat((double *)ap.GetData(), vdim, NbrPoints);
   Array<int> elem_ids(NbrPoints); // el ement ids.
   Array<IntegrationPoint> ips(NbrPoints);  // the location within the element.
   Mesh * m = gf.FESpace()->GetMesh();
   int PointsFounded = m->FindPoints(point_mat, elem_ids, ips); // find the element and the point in the element.
   // get the value of each point one by one.
   for(int i=0; i< NbrPoints; i++)
   {
      av[i] = gf.GetValue(elem_ids[i], ips[i], vdim);
   }
   return PointsFounded;
}


