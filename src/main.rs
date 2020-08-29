use rand::Rng;

fn sigmoid(src: &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
    let mut matrix = src.clone();
      
    for num in matrix.iter_mut().flatten() {
      *num = 1.0_f32 / (1.0_f32 + std::f32::consts::E.powf(-(*num)))
    }
    matrix
}

fn sigmoid_deriv(src: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut matrix: Vec<Vec<f32>> = vec![vec![0.; src[0].len()]; src.len()];
    let out_1 = src.clone();
    let mut out_2 = src.clone();

    out_2.iter_mut().flatten().for_each(|x| *x = 1. - *x);
    
    for row in 0..src.len() {
        for col in 0..src[0].len() {
          matrix[row][col] = out_1[row][col] * out_2[row][col];
        }
    }

    matrix
}

fn multiply(src1: &Vec<Vec<f32>>, src2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut matrix: Vec<Vec<f32>> = vec![vec![0.; src2[0].len()]; src1.len()];

    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for k in 0..src1[0].len() {
                *cell += src1[i][k] * src2[k][j] ;
            }
        }
    }
    matrix
}

fn calculate_error(src1: &Vec<Vec<f32>>, src2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.; src2[0].len()]; src1.len()];

    for row in 0..src1.len() {
        for col in 0..src1[0].len() {
            matrix[row][col] = src1[row][col] - src2[row][col];
        }
    }
    matrix 
}

pub fn transpose(v: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result = vec![vec![0.; v.len()]; v[0].len()];
    
    for i in 0..v.len(){
      for j in 0..v[i].len(){
        result[j][i] = v[i][j];
      }
    }
    result
  }

fn main() {
    let mut rng = rand::thread_rng();

    let input: Vec<Vec<f32>> = vec![vec![0.,0.,1.],
                                    vec![1.,1.,1.],
                                    vec![1.,0.,1.],
                                    vec![0.,1.,1.]];
    
    let output: Vec<Vec<f32>> = vec![vec![0.],
                                      vec![1.],
                                      vec![1.],
                                      vec![0.]];

    let mut l_1 = Vec::new();
    for _i in 0..input[0].len() {
        l_1.push(vec![rng.gen_range(-0.5, 0.5)]);
    }
    
    for i in 0..10000 {
        let output_l1 = sigmoid(&multiply(&input, &l_1));
        let error = calculate_error(&output, &output_l1);

        let adjustment = {
            let mut matrix = vec![vec![0.; error[0].len()]; error.len()];
            let deriv = sigmoid_deriv(&output_l1);
            for row in 0..deriv.len() {
                for col in 0..deriv[0].len() {
                    matrix[row][col] = error[row][col] * deriv[row][col];
                }
            }
            matrix
        };
        
        l_1 = {
            let out = multiply(&transpose(&input), &adjustment);
            let mut matrix = l_1.clone();

            for row in 0..l_1.len() {
                for col in 0..l_1[0].len() {
                    matrix[row][col] = l_1[row][col] + out[row][col];
                }
            }
            matrix
        }; 

        if i == 9999 {
            println!("{:?}", output_l1);
        }
    }
}