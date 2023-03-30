use super::round_to_sum;

#[test]
fn test_round_to_sum() {
    let fractions = vec![2.4, 1.6, 0.8, 1.9, 1.6];
    let steps = vec![1, 2, 4, 8, 1];
    let sum = 20;
    assert_eq!(
        round_to_sum(&fractions, &steps, sum).unwrap(),
        vec![2.0, 2.0, 1.0, 1.0, 2.0]
    );
}
