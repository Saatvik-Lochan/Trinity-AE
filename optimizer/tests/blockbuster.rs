use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use trinity::language::SHAPE_TRACKER;
use trinity::shape::ShapeTracker;
use trinity::*;

fn get_expressions_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("expressions")
}

fn setup_shape_tracker(shapes: Vec<(&str, Vec<usize>)>) {
    SHAPE_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        *tracker = ShapeTracker::new();
        for (name, dims) in shapes {
            tracker.add_tensor(name, dims);
        }
    });
}

#[test]
fn blockbuster_extract_rms_ffn_swiglu_expressions() {
    setup_shape_tracker(vec![
        ("X", vec![12800, 576]),
        ("X_rowsum", vec![12800]),
        ("X_norm", vec![12800, 576]),
        ("W", vec![1536, 576]),
        ("V", vec![1536, 576]),
        ("FF1a", vec![12800, 1536]),
        ("FF1b", vec![12800, 1536]),
        ("FF1a_silu", vec![12800, 1536]),
        ("FF1", vec![12800, 1536]),
        ("O", vec![12800, 576]),
        ("U", vec![576, 1536]),
    ]);
    let expr = "
(seq
    (loop 0 576 tile_k k
        (store (tensor X_rowsum)
            (+
                (x (load (tensor X_rowsum) (index (fulltile))) 1)
                (rsum
                    (sqr (load (input X) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 576 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X_rowsum) (index (fulltile)))
                            576
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 1536 tile_p p
        (loop 0 576 tile_k k
            (store (tensor FF1a)
                (+
                    (x (load (tensor FF1a) (index (fulltile) (tile p))) 1)
                    (*
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input W) (index (tile p) (tile k)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 1536 tile_p p
        (loop 0 576 tile_k k
            (store (tensor FF1b)
                (+
                    (x (load (tensor FF1b) (index (fulltile) (tile p))) 1)
                    (*
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input V) (index (tile p) (tile k)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 1536 tile_p p
        (store (tensor FF1a_silu)
            (x
                (load (tensor FF1a) (index (fulltile) (tile p)))
                (sigmoid
                    (load (tensor FF1a) (index (fulltile) (tile p)))
                )
            )
            (index (fulltile) (tile p))
        )
    )
(seq
    (loop 0 1536 tile_p p
        (store (tensor FF1)
            (x
                (load (tensor FF1a_silu) (index (fulltile) (tile p)))
                (load (tensor FF1b) (index (fulltile) (tile p)))
            )
            (index (fulltile) (tile p))
        )
    )
    (loop 0 576 tile_n n
        (loop 0 1536 tile_p p
            (store (output O)
                (+
                    (x (load (output O) (index (fulltile) (tile n))) 1)
                    (*
                        (load (tensor FF1) (index (fulltile) (tile p)))
                        (load (input U) (index (tile n) (tile p)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
))))))
    ";
    let mut runner = run_until_saturated(expr, rules(), 10);

    let expr_path = get_expressions_path();
    let semi_path =
        expr_path.join("semi/blockbuster_rms_ffn_swiglu_cost6_kern5_wo_scheduler2.json");
    let output_path = expr_path.join("blockbuster_rms_ffn_swiglu_cost6_kern5_wo_scheduler2.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 5) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }

    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(
        &runner,
        semi_path.to_str().unwrap(),
        usize::MAX,
    ) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            println!("{:?}", tile_sets);
            (expressions, tile_sets)
        }
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create(&output_path).expect("Failed to create file");
    let mut writer = BufWriter::new(file);

    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess_v2(expr, &tile_sets);
            format!("{}: {}", i, new_expr)
        })
        .collect::<Vec<String>>()
        .iter()
        .for_each(|line| {
            writeln!(writer, "{}", line).expect("Failed to write to file");
        });

    writer.flush().expect("Failed to flush writer");
}
