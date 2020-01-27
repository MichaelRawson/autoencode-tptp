use glob::glob;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use tptp::parsers::tptp_input_iterator;
use tptp::syntax::*;

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::mem;
use std::path::PathBuf;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum NodeType {
    Defined,
    Variable,
    Functor,
    Argument,
    Application,
    Equality,
    Disequality,
    Negation,
    And,
    Or,
    Equivalent,
    Forall,
    Exists,
    Axiom,
    Conjecture,
    Problem,
}

#[derive(Default)]
struct TPTP {
    save: NodeIndex,
    graph: Graph<NodeType, (), Directed, u32>,
    defined: HashMap<String, NodeIndex>,
    functors: HashMap<String, NodeIndex>,
    variables: HashMap<String, NodeIndex>,
    terms: HashMap<(NodeIndex, Vec<NodeIndex>), NodeIndex>,
    includes: HashMap<PathBuf, Vec<NodeIndex>>,
    inputs: Vec<NodeIndex>,
}

impl TPTP {
    fn node(&mut self, node: NodeType) -> NodeIndex {
        self.graph.add_node(node)
    }

    fn edge(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from, to, ());
    }

    pub fn problem(&mut self, path: &PathBuf) {
        let contents = fs::read(path).unwrap();
        for input in &mut tptp_input_iterator::<()>(contents.as_slice()) {
            self.visit_tptp_input(input);
            self.inputs.push(self.save);
        }
        let problem_index = self.node(NodeType::Problem);
        for input_index in mem::replace(&mut self.inputs, vec![]) {
            self.edge(problem_index, input_index);
        }
    }

    pub fn finish(self) -> Graph<NodeType, (), Directed, u32> {
        self.graph
    }
}

impl Visitor for TPTP {
    fn visit_include(&mut self, include: Include) {
        let path = PathBuf::from(include.file_name);
        if let Some(indices) = self.includes.get(&path) {
            self.inputs.extend_from_slice(indices)
        } else {
            let contents = fs::read(&path).unwrap();
            let mut indices = vec![];
            for input in &mut tptp_input_iterator::<()>(contents.as_slice()) {
                self.visit_tptp_input(input);
                indices.push(self.save);
            }
            self.inputs.extend_from_slice(&indices);
            self.includes.insert(path, indices);
        };
    }

    fn visit_defined_functor(&mut self, defined: DefinedFunctor) {
        let key = format!("{}", defined);
        if let Some(index) = self.defined.get(&key) {
            self.save = *index
        } else {
            let index = self.node(NodeType::Defined);
            self.defined.insert(key, index);
            self.save = index;
        }
    }

    fn visit_variable(&mut self, variable: Variable) {
        let key = format!("{}", variable);
        if let Some(index) = self.variables.get(&key) {
            self.save = *index
        } else {
            let index = self.node(NodeType::Variable);
            self.variables.insert(key, index);
            self.save = index;
        }
    }

    fn visit_functor(&mut self, functor: Functor) {
        let key = format!("{}", functor);
        if let Some(index) = self.functors.get(&key) {
            self.save = *index;
        } else {
            let index = self.node(NodeType::Functor);
            self.functors.insert(key, index);
            self.save = index;
        }
    }

    fn visit_fof_plain_term(&mut self, term: FofPlainTerm) {
        match term {
            FofPlainTerm::Constant(functor) => self.visit_functor(functor),
            FofPlainTerm::Function(functor, arguments) => {
                self.visit_functor(functor);
                let functor_index = self.save;

                let mut argument_indices = vec![];
                for argument in arguments.0 {
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

                    let mut auxiliary_indices = vec![];
                    for argument_index in &argument_indices {
                        let auxiliary_index = self.node(NodeType::Argument);
                        self.edge(auxiliary_index, *argument_index);
                        self.edge(application_index, auxiliary_index);
                        auxiliary_indices.push(auxiliary_index);
                    }
                    for i in 1..auxiliary_indices.len() {
                        self.edge(
                            auxiliary_indices[i - 1],
                            auxiliary_indices[i],
                        );
                    }

                    let key = (functor_index, argument_indices);
                    self.terms.insert(key, application_index);
                    self.save = application_index
                }
            }
        }
    }

    fn visit_fof_defined_infix_formula(
        &mut self,
        infix: FofDefinedInfixFormula,
    ) {
        let equality_index = self.node(NodeType::Equality);
        self.visit_fof_term(infix.left);
        self.edge(equality_index, self.save);
        self.visit_fof_term(infix.right);
        self.edge(equality_index, self.save);
        self.save = equality_index
    }

    fn visit_fof_infix_unary(&mut self, infix: FofInfixUnary) {
        let disequality_index = self.node(NodeType::Disequality);
        self.visit_fof_term(infix.left);
        self.edge(disequality_index, self.save);
        self.visit_fof_term(infix.right);
        self.edge(disequality_index, self.save);
        self.save = disequality_index;
    }

    fn visit_fof_unary_formula(&mut self, unary: FofUnaryFormula) {
        use FofUnaryFormula::*;
        match unary {
            Unary(_, unit) => {
                let negation_index = self.node(NodeType::Negation);
                self.visit_fof_unit_formula(unit);
                self.edge(negation_index, self.save);
                self.save = negation_index;
            }
            InfixUnary(infix) => self.visit_fof_infix_unary(infix),
        }
    }

    fn visit_fof_quantified_formula(
        &mut self,
        quantified: FofQuantifiedFormula,
    ) {
        use FofQuantifier::*;
        let node_type = match quantified.quantifier {
            Forall => NodeType::Forall,
            Exists => NodeType::Exists,
        };
        let quantifier_index = self.node(node_type);

        let mut restore = vec![];
        for variable in &quantified.bound.0 {
            let key = format!("{}", variable);
            if let Some((clash, index)) = self.variables.remove_entry(&key) {
                restore.push((clash, index));
            }

            let variable_index = self.node(NodeType::Variable);
            self.variables.insert(key, variable_index);
            self.edge(quantifier_index, variable_index);
        }

        self.visit_fof_unit_formula(quantified.formula);
        self.edge(quantifier_index, self.save);

        for variable in &quantified.bound.0 {
            self.variables.remove(&format!("{}", variable));
        }
        for (clash, index) in restore {
            self.variables.insert(clash, index);
        }

        self.save = quantifier_index;
    }

    fn visit_fof_binary_nonassoc(&mut self, nonassoc: FofBinaryNonassoc) {
        use NonassocConnective::*;

        self.visit_fof_unit_formula(nonassoc.left);
        let left_index = self.save;
        self.visit_fof_unit_formula(nonassoc.right);
        let right_index = self.save;

        match nonassoc.op {
            LRImplies => {
                let negated_index = self.node(NodeType::Negation);
                self.edge(negated_index, left_index);
                let or_index = self.node(NodeType::Or);
                self.edge(or_index, negated_index);
                self.edge(or_index, right_index);
                self.save = or_index;
            }
            RLImplies => {
                let negated_index = self.node(NodeType::Negation);
                self.edge(negated_index, right_index);
                let or_index = self.node(NodeType::Or);
                self.edge(or_index, negated_index);
                self.edge(or_index, left_index);
                self.save = or_index;
            }
            Equivalent => {
                let equiv_index = self.node(NodeType::Equivalent);
                self.edge(equiv_index, left_index);
                self.edge(equiv_index, right_index);
                self.save = equiv_index;
            }
            NotEquivalent => {
                let equiv_index = self.node(NodeType::Equivalent);
                let negated_index = self.node(NodeType::Negation);
                self.edge(equiv_index, left_index);
                self.edge(equiv_index, right_index);
                self.edge(negated_index, equiv_index);
                self.save = negated_index;
            }
            NotAnd => {
                let and_index = self.node(NodeType::And);
                let negated_index = self.node(NodeType::Negation);
                self.edge(and_index, left_index);
                self.edge(and_index, right_index);
                self.edge(negated_index, and_index);
                self.save = negated_index;
            }
            NotOr => {
                let or_index = self.node(NodeType::Or);
                let negated_index = self.node(NodeType::Negation);
                self.edge(or_index, left_index);
                self.edge(or_index, right_index);
                self.edge(negated_index, or_index);
                self.save = negated_index;
            }
        }
    }

    fn visit_fof_or_formula(&mut self, or: FofOrFormula) {
        let or_index = self.node(NodeType::Or);
        for child in or.0 {
            self.visit_fof_unit_formula(child);
            self.edge(or_index, self.save);
        }
        self.save = or_index;
    }

    fn visit_fof_and_formula(&mut self, and: FofAndFormula) {
        let and_index = self.node(NodeType::And);
        for child in and.0 {
            self.visit_fof_unit_formula(child);
            self.edge(and_index, self.save);
        }
        self.save = and_index;
    }

    fn visit_fof_annotated(&mut self, annotated: FofAnnotated) {
        self.variables.clear();
        self.visit_fof_formula(annotated.formula);
        use FormulaRole::*;
        let node_type = match annotated.role {
            Axiom | Definition | Hypothesis | Lemma => NodeType::Axiom,
            FormulaRole::Conjecture => NodeType::Conjecture,
            _ => unreachable!(),
        };
        let input_index = self.node(node_type);
        self.edge(input_index, self.save);
        self.save = input_index;
    }
}

fn main() {
    let mut tptp = TPTP::default();
    for path in glob("**/*+*.p").unwrap() {
        let path = path.unwrap();
        eprintln!("{:?}", path);
        if !(path.starts_with("Problems/SYN/")
            || path.starts_with("Problems/SYO/")
            || path.starts_with("Problems/CSR/")
            || path.starts_with("Problems/HWV/")
            || path.starts_with("Problems/SWV/")
            || path.starts_with("Problems/SWW/"))
        {
            tptp.problem(&path);
        }
    }
    let graph = tptp.finish();
    eprintln!("{} nodes, {} edges", graph.node_count(), graph.edge_count());
    let (nodes, edges) = graph.into_nodes_edges();

    {
        eprintln!("writing nodes.dat...");
        let bytes: Vec<u8> =
            nodes.into_iter().map(|node| node.weight as u8).collect();
        let mut node_file = fs::File::create("nodes.dat").unwrap();
        node_file.write_all(&bytes).unwrap();
    }
    {
        eprintln!("writing edges.dat...");
        let mut from = vec![];
        let mut to = vec![];
        for edge in edges {
            from.push(edge.source().index() as i64);
            to.push(edge.target().index() as i64);
        }
        let from_bytes = unsafe {
            std::slice::from_raw_parts(
                from.as_ptr() as *const u8,
                from.len() * std::mem::size_of::<i64>(),
            )
        };
        let to_bytes = unsafe {
            std::slice::from_raw_parts(
                to.as_ptr() as *const u8,
                to.len() * std::mem::size_of::<i64>(),
            )
        };
        let mut edge_file = fs::File::create("edges.dat").unwrap();
        edge_file.write_all(&from_bytes).unwrap();
        edge_file.write_all(&to_bytes).unwrap();
    }
    eprintln!("OK")
}
