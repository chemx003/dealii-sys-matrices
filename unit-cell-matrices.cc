#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/base/point.h>
#include <fstream>
#include <iostream>
namespace unitcell
{
	//  Class declaration
	using namespace dealii;
	template<int dim>
	class MatrixCalc
	{
	public:
		MatrixCalc ();
		~MatrixCalc ();
		void run ();
	private:
		void setup_system ();
		void assemble_system ();
		void output_results ();
		Triangulation<dim> 		triangulation;
		DoFHandler<dim>			dof_handler;
		FESystem<dim>			fe;
		MappingQ<dim,dim>		mapping;
		SparsityPattern			stiffness_sp;
		SparsityPattern			mass_sp;
		SparseMatrix<double>	stiffness_matrix;
		SparseMatrix<double>	mass_matrix;
	};

	//  Contructor
	template <int dim>
	MatrixCalc<dim>::MatrixCalc ()
		:
		dof_handler(triangulation),
		fe (FE_Q<dim>(1), dim),
		mapping(dim, true)
	{}

	template <int dim>
	MatrixCalc<dim>::~MatrixCalc ()
	{
		dof_handler.clear ();
	}

	template <int dim>
	void MatrixCalc<dim>::setup_system ()
	{
		dof_handler.distribute_dofs (fe);

		DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler, dsp);

		stiffness_sp.copy_from(dsp);
		mass_sp.copy_from(dsp);

		stiffness_matrix.reinit(stiffness_sp);
		mass_matrix.reinit(mass_sp);
	}

	template <int dim>
	void MatrixCalc<dim>::assemble_system ()
	{
		QGauss<dim> quadrature_formula(2);
		FEValues<dim> fe_values (fe, quadrature_formula,
								update_values | update_gradients |
								update_quadrature_points | update_JxW_values);
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points	 = quadrature_formula.size();
		FullMatrix<double>	cell_kmatrix (dofs_per_cell, dofs_per_cell);
		FullMatrix<double>  cell_mmatrix (dofs_per_cell, dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
		std::vector<double>	youngs_values (n_q_points);
		std::vector<double> poisson_values (n_q_points);
		ConstantFunction<dim> youngs(1), poisson(0.5);

		typename DoFHandler<dim>::active_cell_iterator 
			cell = dof_handler.begin_active(),
			endc = dof_handler.end();

		for(; cell!=endc; ++cell)
		{
			cell_kmatrix = 0;
			cell_mmatrix = 0;
			fe_values.reinit (cell);
			youngs.value_list
				(fe_values.get_quadrature_points(),youngs_values);
			poisson.value_list
				(fe_values.get_quadrature_points(),poisson_values);

			for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
				const unsigned int
				component_i = fe.system_to_component_index(i).first;

				for(unsigned int j=0; j<dofs_per_cell; ++j)
				{
					const unsigned int
					component_j = fe.system_to_component_index(j).first;

					for(unsigned int q_point=0; q_point<n_q_points;
							++q_point)
					{
						cell_kmatrix(i,j)+=(
						(fe_values.shape_grad(i,q_point)[component_i] *
						 fe_values.shape_grad(j,q_point)[component_j] *
						 (poisson_values[q_point]*youngs_values[q_point]) /
						 (1 - poisson_values[q_point]*poisson_values[q_point]))
						+
						(fe_values.shape_grad(i,q_point)[component_j] *
						 fe_values.shape_grad(j,q_point)[component_i] *
						 (youngs_values[q_point]) /
						 (1 + poisson_values[q_point]))
						+
						((component_i==component_j) ?
						 ((fe_values.shape_grad(i,q_point)[component_j] *
						 fe_values.shape_grad(j,q_point)[component_i] 
						 +
						 fe_values.shape_grad(i,q_point) *
						 fe_values.shape_grad(j,q_point)) *
						 (youngs_values[q_point]) /
						 (1 + poisson_values[q_point]))
						 : 0)
						)*fe_values.JxW(q_point);

						cell_mmatrix(i,j)+=(
						fe_values.shape_value(i,q_point) *
						fe_values.shape_value(j,q_point)
						)*fe_values.JxW(q_point);
					}
				}
			}

			cell->get_dof_indices (local_dof_indices);
			for(unsigned int i=0; i<dofs_per_cell; ++i)
			{
				for(unsigned int j=0; j<dofs_per_cell; ++j)
				{
					stiffness_matrix.add (local_dof_indices[i],
									   local_dof_indices[j],
									   cell_kmatrix(i,j));

					mass_matrix.add (local_dof_indices[i],
									local_dof_indices[j],
									cell_mmatrix(i,j));
				}
			}
		}
	}

	template<int dim>
	void MatrixCalc<dim>::output_results ()
	{
		//  Print stiffness matrix
		std::string filename = "kmatrix";
		filename +=".txt";
		
		std::ofstream output (filename);
		stiffness_matrix.print_formatted(output,6,false, 0, "0", 1.0);
		output.close();

		//  Print mass matrix
		std::string filename2 = "mmatrix";
		filename2 +=".txt";

		std::ofstream output2 (filename2);
		mass_matrix.print_formatted(output2,6,false,0,"0",1.0);
		output2.close();

		//  Print DoFs and locations
		std::string filename3 = "dof-locations";
		filename3 +=".txt";
		std::ofstream output3 (filename3);

		std::vector<Point<dim>> support_points (dof_handler.n_dofs());
		DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

		for(unsigned int i=0; i<dof_handler.n_dofs(); i++)
		{
			output3 	<< i+1 << "\t" 
						<< (support_points[i])(0) << "\t"
						<< (support_points[i])(1) << "\n";
		}
		output3.close();
		
		//  Sort the DOFs
		//  check interior
		//  bot
		//  top
		//  right
		//  left
		//  lb
		//  rb
		//  lt
		//  rt
	}

	template<int dim>
	void MatrixCalc<dim>::run ()
	{
		GridGenerator::hyper_cube(triangulation, -1, 1);
		triangulation.refine_global (2);

	std::cout 	<< "    Number of active cells:    "
				<< triangulation.n_active_cells()
				<< std::endl;

	setup_system ();

	std::cout 	<< "    Number of degrees of freedom:  "
				<< dof_handler.n_dofs()
				<< std::endl;

	assemble_system ();
	output_results();
	}
}

int main ()
{
	try
	{
		unitcell::MatrixCalc<2> MatrixCalc;
		MatrixCalc.run ();
	}
	catch (std::exception &exc)
	{
		std::cerr 	<< std::endl << std::endl
					<<"-------------------------------------------------------"
					<< std::endl;
		std::cerr	<< "Exception on processing: " << std::endl
					<< exc.what() << std::endl
					<< "Aborting!" << std::endl
					<<"-------------------------------------------------------"
					<< std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr 	<< std::endl << std::endl
					<<"-------------------------------------------------------"
					<< std::endl;
		std::cerr 	<< "Unknown exeption!" << std::endl
					<< "Aborting!" << std::endl
					<<"-------------------------------------------------------"
					<< std::endl;
		return 1;
	}

	return 0;
}
