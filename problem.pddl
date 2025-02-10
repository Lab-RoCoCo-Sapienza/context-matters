(define (problem laundry_task)
    (:domain allensville)
    (:objects
        washing_machine - location
        laundry_basket - location
        clothes_package - package
        delivery_vehicle - vehicle
    )
    (:init
        (at laundry_basket)
        (not (delivered clothes_package))
    )
    (:goal
        (and
            (at washing_machine)
            (delivered clothes_package)
        )
    )
)