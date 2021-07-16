use fnv::FnvHashMap as HashMap;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use std::fs;
use std::mem;
use std::path::Path;
use tptp::cnf::{Disjunction, Literal};
use tptp::common::*;
use tptp::fof::*;
use tptp::top::*;
use tptp::visitor::Visitor;
use tptp::TPTPIterator;

#[repr(i64)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum NodeType {
    Defined,
    Variable,
    Functor,
    Application,
    Equality,
    Negation,
    And,
    Or,
    Equivalent,
    Forall,
    Exists,
    Axiom,
    Conjecture,
}

type NodeIndex = i64;

#[derive(Default)]
struct TPTP {
    save: NodeIndex,
    nodes: Vec<NodeType>,
    from: Vec<NodeIndex>,
    to: Vec<NodeIndex>,
    defined: HashMap<String, NodeIndex>,
    functors: HashMap<String, NodeIndex>,
    variables: HashMap<String, NodeIndex>,
    terms: HashMap<(NodeIndex, Vec<NodeIndex>), NodeIndex>,
    equalities: HashMap<(NodeIndex, NodeIndex), NodeIndex>,
    negations: HashMap<NodeIndex, NodeIndex>,
    disjunctions: HashMap<Vec<NodeIndex>, NodeIndex>,
    conjunctions: HashMap<Vec<NodeIndex>, NodeIndex>,
    equivalences: HashMap<(NodeIndex, NodeIndex), NodeIndex>,
    inputs: Vec<NodeIndex>,
}

impl TPTP {
    fn node(&mut self, node: NodeType) -> NodeIndex {
        let index = self.nodes.len() as NodeIndex;
        self.nodes.push(node);
        index
    }

    fn edge(&mut self, from: NodeIndex, to: NodeIndex) {
        self.from.push(from);
        self.to.push(to);
    }

    fn equation(&mut self, left: &Term, right: &Term) {
        self.visit_fof_term(left);
        let left = self.save;
        self.visit_fof_term(right);
        let right = self.save;

        if let Some(equality_index) = self.equalities.get(&(left, right)) {
            self.save = *equality_index;
        } else {
            let equality_index = self.node(NodeType::Equality);
            self.edge(equality_index, left);
            self.edge(equality_index, right);
            self.equalities.insert((left, right), equality_index);
            self.save = equality_index;
        }
    }

    fn negation(&mut self) {
        if let Some(negation_index) = self.negations.get(&self.save) {
            self.save = *negation_index;
        } else {
            let negation_index = self.node(NodeType::Negation);
            self.edge(negation_index, self.save);
            self.negations.insert(self.save, negation_index);
            self.save = negation_index;
        }
    }

    fn equivalence(&mut self, mut left: NodeIndex, mut right: NodeIndex) {
        if left > right {
            mem::swap(&mut left, &mut right);
        }
        if let Some(index) = self.equivalences.get(&(left, right)) {
            self.save = *index;
        } else {
            let equiv_index = self.node(NodeType::Equivalent);
            self.edge(equiv_index, left);
            self.edge(equiv_index, right);
            self.save = equiv_index;
        }
    }

    fn or(&mut self, mut indices: Vec<NodeIndex>) {
        indices.sort_unstable();
        indices.dedup();
        if let Some(or_index) = self.disjunctions.get(&indices) {
            self.save = *or_index;
        } else {
            let or_index = self.node(NodeType::Or);
            for index in indices {
                self.edge(or_index, index);
            }
            self.save = or_index;
        }
    }

    fn and(&mut self, mut indices: Vec<NodeIndex>) {
        indices.sort_unstable();
        indices.dedup();
        if let Some(and_index) = self.conjunctions.get(&indices) {
            self.save = *and_index;
        } else {
            let and_index = self.node(NodeType::And);
            for index in indices {
                self.edge(and_index, index);
            }
            self.save = and_index;
        }
    }

    pub(crate) fn file(&mut self, path: &Path) {
        let contents = fs::read(path).expect("I/O error");
        let iterator = TPTPIterator::<()>::new(contents.as_slice());
        for input in iterator {
            self.visit_tptp_input(&input.expect("parse error"));
        }
    }

    pub(crate) fn finish(
        self,
    ) -> (Vec<NodeType>, Vec<NodeIndex>, Vec<NodeIndex>) {
        (self.nodes, self.from, self.to)
    }
}

impl Visitor<'_> for TPTP {
    fn visit_include(&mut self, include: &Include) {
        let path = Path::new(include.file_name.0 .0);
        self.file(path);
    }

    fn visit_defined_functor(&mut self, defined: &DefinedFunctor) {
        let key = format!("{}", defined);
        if let Some(index) = self.defined.get(&key) {
            self.save = *index
        } else {
            let index = self.node(NodeType::Defined);
            self.defined.insert(key, index);
            self.save = index;
        }
    }

    fn visit_variable(&mut self, variable: &Variable) {
        let key = format!("{}", variable);
        if let Some(index) = self.variables.get(&key) {
            self.save = *index
        } else {
            let index = self.node(NodeType::Variable);
            self.variables.insert(key, index);
            self.save = index;
        }
    }

    fn visit_functor(&mut self, functor: &Functor) {
        let key = format!("{}", functor);
        if let Some(index) = self.functors.get(&key) {
            self.save = *index;
        } else {
            let index = self.node(NodeType::Functor);
            self.functors.insert(key, index);
            self.save = index;
        }
    }

    fn visit_fof_plain_term(&mut self, term: &PlainTerm) {
        match term {
            PlainTerm::Constant(functor) => self.visit_functor(&functor.0),
            PlainTerm::Function(functor, arguments) => {
                self.visit_functor(&functor);
                let functor_index = self.save;

                let mut argument_indices = vec![];
                for argument in &arguments.0 {
                    self.visit_fof_term(argument);
                    argument_indices.push(self.save);
                }

                let key = (functor_index, argument_indices);
                if let Some(index) = self.terms.get(&key) {
                    self.save = *index;
                } else {
                    let (functor_index, argument_indices) = key;
                    let application_index = self.node(NodeType::Application);
                    self.edge(application_index, functor_index);
                    for argument_index in &argument_indices {
                        self.edge(application_index, *argument_index);
                    }

                    let key = (functor_index, argument_indices);
                    self.terms.insert(key, application_index);
                    self.save = application_index
                }
            }
        }
    }

    fn visit_fof_defined_infix_formula(&mut self, infix: &DefinedInfixFormula) {
        self.equation(&infix.left, &infix.right);
    }

    fn visit_fof_unary_formula(&mut self, unary: &UnaryFormula) {
        use UnaryFormula::*;
        match unary {
            Unary(_, unit) => {
                self.visit_fof_unit_formula(&unit);
            }
            InfixUnary(infix) => {
                self.equation(&infix.left, &infix.right);
            }
        }
        self.negation();
    }

    fn visit_fof_quantified_formula(&mut self, quantified: &QuantifiedFormula) {
        use Quantifier::*;
        let node_type = match quantified.quantifier {
            Forall => NodeType::Forall,
            Exists => NodeType::Exists,
        };

        let mut variable_indices = vec![];
        let mut restore = vec![];
        for variable in &quantified.bound.0 {
            let key = format!("{}", variable);
            if let Some((clash, index)) = self.variables.remove_entry(&key) {
                restore.push((clash, index));
            }

            let variable_index = self.node(NodeType::Variable);
            variable_indices.push(variable_index);
            self.variables.insert(key, variable_index);
        }
        self.visit_fof_unit_formula(&quantified.formula);

        let quantifier_index = self.node(node_type);
        self.edge(quantifier_index, self.save);
        for variable_index in variable_indices {
            self.edge(quantifier_index, variable_index);
        }

        for variable in &quantified.bound.0 {
            self.variables.remove(&format!("{}", variable));
        }
        for (clash, index) in restore {
            self.variables.insert(clash, index);
        }

        self.save = quantifier_index;
    }

    fn visit_fof_binary_nonassoc(&mut self, nonassoc: &BinaryNonassoc) {
        use NonassocConnective::*;

        self.visit_fof_unit_formula(&nonassoc.left);
        let left_index = self.save;
        self.visit_fof_unit_formula(&nonassoc.right);
        let right_index = self.save;

        match nonassoc.op {
            LRImplies => {
                self.save = left_index;
                self.negation();
                self.or(vec![self.save, right_index]);
            }
            RLImplies => {
                self.save = right_index;
                self.negation();
                self.or(vec![left_index, self.save]);
            }
            Equivalent => {
                self.equivalence(left_index, right_index);
            }
            NotEquivalent => {
                self.equivalence(left_index, right_index);
                self.negation();
            }
            NotAnd => {
                self.and(vec![left_index, right_index]);
                self.negation();
            }
            NotOr => {
                self.or(vec![left_index, right_index]);
                self.negation();
            }
        }
    }

    fn visit_fof_or_formula(&mut self, or: &OrFormula) {
        let mut child_indices = vec![];
        for child in &or.0 {
            self.visit_fof_unit_formula(child);
            child_indices.push(self.save);
        }
        self.or(child_indices);
    }

    fn visit_fof_and_formula(&mut self, and: &AndFormula) {
        let mut child_indices = vec![];
        for child in &and.0 {
            self.visit_fof_unit_formula(child);
            child_indices.push(self.save);
        }
        self.and(child_indices);
    }

    fn visit_literal(&mut self, literal: &Literal) {
        use Literal::*;
        match literal {
            Atomic(f) => {
                self.visit_fof_atomic_formula(f);
            }
            NegatedAtomic(f) => {
                self.visit_fof_atomic_formula(f);
                self.negation();
            }
            Infix(infix) => {
                self.equation(&infix.left, &infix.right);
                self.negation();
            }
        }
    }

    fn visit_disjunction(&mut self, or: &Disjunction) {
        let mut child_indices = vec![];
        for child in &or.0 {
            self.visit_literal(child);
            child_indices.push(self.save);
        }
        if child_indices.len() > 1 {
            self.or(child_indices);
        }
    }

    fn visit_fof_annotated(&mut self, annotated: &FofAnnotated) {
        self.variables.clear();
        self.visit_fof_formula(&annotated.0.formula);
        let node_type = match annotated.0.role.0 .0 {
            "conjecture" => NodeType::Conjecture,
            _ => NodeType::Axiom,
        };
        let input_index = self.node(node_type);
        self.edge(input_index, self.save);
        self.inputs.push(input_index);
        self.save = input_index;
    }

    fn visit_cnf_annotated(&mut self, annotated: &CnfAnnotated) {
        self.variables.clear();
        self.visit_cnf_formula(&annotated.0.formula);
        let node_type = match annotated.0.role.0 .0 {
            "negated_conjecture" => NodeType::Conjecture,
            _ => NodeType::Axiom,
        };
        let input_index = self.node(node_type);
        self.edge(input_index, self.save);
        self.inputs.push(input_index);
        self.save = input_index;
    }
}

#[pymodule]
fn tptp_graph(_python: Python, module: &PyModule) -> PyResult<()> {
    #[pyfn(module)]
    fn graph_of<'py>(
        python: Python<'py>,
        path: &str,
    ) -> PyResult<(&'py PyArray1<i64>, &'py PyArray1<i64>, &'py PyArray1<i64>)>
    {
        let mut tptp = TPTP::default();
        tptp.file(Path::new(path));
        let (nodes, sources, targets) = tptp.finish();

        let node_ptr = nodes.as_ptr() as *const NodeType as *const i64;
        let nodes =
            unsafe { std::slice::from_raw_parts(node_ptr, nodes.len()) };
        let nodes = nodes.to_pyarray(python);
        let sources = sources.to_pyarray(python);
        let targets = targets.to_pyarray(python);
        Ok((nodes, sources, targets))
    }

    Ok(())
}
